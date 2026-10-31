#include "qwen35_dflash/tokenizer.hpp"
#include "oniguruma.h"
#include "utf8proc.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>

namespace qwen35::dflash {
namespace {
void Check(bool v, const char* error) { if (!v) throw std::runtime_error(error); }
constexpr std::uint32_t Missing = UINT32_MAX;
struct Reader {
  std::ifstream f;
  explicit Reader(const std::string& path):f(path, std::ios::binary) {
    Check(bool(f), "tokenizer bundle open failed");
    char h[8]; f.read(h,8); Check(f && std::string(h,8)=="QWTOK001", "tokenizer bundle header mismatch");
  }
  std::uint32_t U32() {
    unsigned char b[4]; f.read(reinterpret_cast<char*>(b),4); Check(bool(f), "tokenizer bundle truncated");
    return b[0] | std::uint32_t(b[1])<<8 | std::uint32_t(b[2])<<16 | std::uint32_t(b[3])<<24;
  }
  std::string Blob() {
    auto n=U32(); Check(n && n<=16384, "tokenizer string extent invalid");
    std::string s(n,'\0'); f.read(s.data(),n); Check(bool(f), "tokenizer string truncated"); return s;
  }
};
std::uint64_t Pair(std::uint32_t a,std::uint32_t b) { return std::uint64_t(a)<<32 | b; }
bool Space(std::int32_t c) {
  // Python str.strip (used by the pinned Jinja template), including U+001C..1F.
  return (c>=9 && c<=13) || (c>=28 && c<=32) || c==0x85 || c==0xa0 || c==0x1680 ||
         (c>=0x2000 && c<=0x200a) || c==0x2028 || c==0x2029 || c==0x202f || c==0x205f || c==0x3000;
}
std::string Trim(const std::string& s) {
  std::size_t start=0,end=0,at=0; bool started=false;
  while(at<s.size()) {
    utf8proc_int32_t c=0; auto n=utf8proc_iterate(reinterpret_cast<const unsigned char*>(s.data()+at),s.size()-at,&c);
    Check(n>0 && c!=0,"prompt must be valid UTF-8 without NUL");
    if (!Space(c)) { if(!started) start=at; started=true; end=at+n; } at+=n;
  }
  return started?s.substr(start,end-start):"";
}
std::string NFC(const std::string& s) {
  utf8proc_uint8_t* result=nullptr;
  auto n=utf8proc_map(reinterpret_cast<const unsigned char*>(s.data()),s.size(),&result,
      static_cast<utf8proc_option_t>(UTF8PROC_STABLE|UTF8PROC_COMPOSE));
  Check(n>=0,"tokenizer NFC normalization failed");
  std::unique_ptr<unsigned char,decltype(&std::free)> owned(result,&std::free);
  return std::string(reinterpret_cast<const char*>(result),n);
}
std::string LossyUTF8(const std::string& s) {
  std::string out;
  for(std::size_t i=0;i<s.size();) {
    unsigned char a=s[i]; std::size_t count=a<128?1:a>=0xc2&&a<=0xdf?2:a>=0xe0&&a<=0xef?3:a>=0xf0&&a<=0xf4?4:0;
    if(!count) {out+="\xef\xbf\xbd";++i;continue;}
    std::size_t good=1;
    while(good<count && i+good<s.size()) {
      unsigned char b=s[i+good];
      if(b<0x80 || b>0xbf || (good==1 && ((a==0xe0&&b<0xa0)||(a==0xed&&b>0x9f)||(a==0xf0&&b<0x90)||(a==0xf4&&b>0x8f))))break;
      ++good;
    }
    if(good==count)out.append(s,i,count);else out+="\xef\xbf\xbd";
    i+=good;
  }
  return out;
}
}

struct TextTokenizer::Impl {
  struct Added {std::uint32_t id;bool special;std::string text;};
  struct Merge {std::uint32_t rank,id;};
  std::vector<std::string> vocab;
  std::array<std::uint32_t,256> bytes;
  std::vector<Added> added;
  std::unordered_map<std::uint64_t,Merge> merges;
  OnigRegex regex=nullptr;
  ~Impl(){if(regex)onig_free(regex);}
  explicit Impl(const std::string& path) {
    Reader r(path);auto pattern=r.Blob(); auto n=r.U32();Check(n==248044,"tokenizer must use original 248044-entry base vocabulary");
    bytes.fill(Missing);vocab.reserve(n+26);
    for(std::uint32_t i=0;i<n;++i){auto s=r.Blob();if(s.size()==1)bytes[static_cast<unsigned char>(s[0])]=i;vocab.push_back(std::move(s));}
    for(auto id:bytes)Check(id!=Missing,"tokenizer missing byte token");
    auto count=r.U32();Check(count==26,"tokenizer added-token count mismatch");
    for(std::uint32_t i=0;i<count;++i){auto id=r.U32(),flag=r.U32();auto s=r.Blob();Check(id==vocab.size()&&flag<=1,"added token ID/flag invalid");vocab.push_back(s);added.push_back({id,bool(flag),std::move(s)});}
    count=r.U32();Check(count==247587,"tokenizer merge count mismatch");merges.reserve(count);
    for(std::uint32_t rank=0;rank<count;++rank){auto a=r.U32(),b=r.U32(),id=r.U32();Check(a<n&&b<n&&id<n&&vocab[a]+vocab[b]==vocab[id],"BPE merge does not bind vocabulary bytes");Check(merges.emplace(Pair(a,b),Merge{rank,id}).second,"duplicate BPE merge");}
    Check(r.f.peek()==std::char_traits<char>::eof(),"trailing tokenizer bundle data");
    OnigEncoding enc=ONIG_ENCODING_UTF8;Check(onig_initialize(&enc,1)==ONIG_NORMAL,"Oniguruma init failed");
    OnigErrorInfo error{};
    int status=onig_new(&regex,reinterpret_cast<const unsigned char*>(pattern.data()),reinterpret_cast<const unsigned char*>(pattern.data()+pattern.size()),ONIG_OPTION_NONE,enc,ONIG_SYNTAX_RUBY,&error);
    Check(status==ONIG_NORMAL,"tokenizer Unicode regex compilation failed");
  }
  void BPE(const std::string& s,std::vector<std::uint32_t>& out) const {
    if(s.empty())return;
    struct Node{std::uint32_t id;int prev,next;};
    struct Candidate{std::uint32_t rank,id,left_id,right_id;int left,right;bool operator<(const Candidate& b)const{return rank!=b.rank?rank>b.rank:left>b.left;}};
    std::vector<Node> nodes;nodes.reserve(s.size());for(std::size_t i=0;i<s.size();++i)nodes.push_back({bytes[static_cast<unsigned char>(s[i])],int(i)-1,i+1<s.size()?int(i+1):-1});
    std::priority_queue<Candidate> heap;
    auto push=[&](int i){if(i<0)return;auto& a=nodes[i];if(a.id==Missing||a.next<0)return;auto& b=nodes[a.next];auto it=merges.find(Pair(a.id,b.id));if(it!=merges.end())heap.push({it->second.rank,it->second.id,a.id,b.id,i,a.next});};
    for(std::size_t i=0;i<nodes.size();++i)push(i);
    while(!heap.empty()) {
      auto m=heap.top();heap.pop();auto& a=nodes[m.left];auto& b=nodes[m.right];
      if(a.id!=m.left_id||b.id!=m.right_id||a.next!=m.right)continue;
      a.id=m.id;a.next=b.next;b.id=Missing;if(a.next>=0)nodes[a.next].prev=m.left;push(a.prev);push(m.left);
    }
    for(int i=0;i>=0;i=nodes[i].next)out.push_back(nodes[i].id);
  }
  void Ordinary(const std::string& input,std::vector<std::uint32_t>& out)const {
    if(input.empty())return;auto s=NFC(input);
    std::unique_ptr<OnigRegion,void(*)(OnigRegion*)> region(onig_region_new(),[](OnigRegion* r){onig_region_free(r,1);});
    Check(bool(region),"regex region allocation failed");
    auto begin=reinterpret_cast<const unsigned char*>(s.data());auto end=begin+s.size();std::size_t pos=0;
    while(pos<s.size()) {
      auto result=onig_search(regex,begin,end,begin+pos,end,region.get(),ONIG_OPTION_NONE);
      if(result==ONIG_MISMATCH){BPE(s.substr(pos),out);break;}
      Check(result>=0&&region->end[0]>region->beg[0],"tokenizer Unicode regex match failed");
      if(std::size_t(region->beg[0])>pos)BPE(s.substr(pos,region->beg[0]-pos),out);
      BPE(s.substr(region->beg[0],region->end[0]-region->beg[0]),out);pos=region->end[0];
    }
  }
};
TextTokenizer::TextTokenizer(const std::string& bundle):impl_(std::make_unique<Impl>(bundle)){}
TextTokenizer::~TextTokenizer()=default;
std::vector<std::uint32_t> TextTokenizer::Encode(const std::string& text,const std::string& mode)const {
  Check(!text.empty()&&text.size()<=16384,"prompt text exceeds 16KiB or is empty");
  auto trimmed=Trim(text);std::string s;
  if(mode=="raw")s=text;
  else {Check(mode=="chat","unknown text prompt format");
    Check(!(trimmed.rfind("<tool_response>",0)==0 && trimmed.size()>=16 && trimmed.compare(trimmed.size()-16,16,"</tool_response>")==0),"chat requires a user query, not only a tool response");
    s="<|im_start|>user\n"+trimmed+"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";}
  std::vector<std::uint32_t> out;std::size_t pos=0;
  while(pos<s.size()){
    std::size_t best=s.size();const Impl::Added* chosen=nullptr;
    for(const auto& a:impl_->added){auto at=s.find(a.text,pos);if(at<best||(at==best&&at!=std::string::npos&&chosen&&a.text.size()>chosen->text.size())){best=at;chosen=&a;}}
    impl_->Ordinary(s.substr(pos,best-pos),out);if(!chosen)break;out.push_back(chosen->id);pos=best+chosen->text.size();
  }
  return out;
}
std::string TextTokenizer::Decode(const std::vector<std::uint32_t>& ids,bool skip_special)const {
  std::string bytes;
  for(auto id:ids){if(id>=impl_->vocab.size()){bytes+="\xef\xbf\xbd";continue;}
    bool special=false;for(const auto& a:impl_->added)if(a.id==id)special=a.special;
    if(!skip_special||!special)bytes+=impl_->vocab[id];}
  return LossyUTF8(bytes);
}
std::vector<std::uint32_t> TextTokenizer::Unmapped(const std::vector<std::uint32_t>& ids)const {
  std::vector<std::uint32_t> result;for(auto id:ids)if(id>=impl_->vocab.size())result.push_back(id);return result;
}
}
