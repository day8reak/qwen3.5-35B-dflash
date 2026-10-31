#ifdef QWEN35_TEXT_TOKENIZER
#include "qwen35_dflash/tokenizer.hpp"
#endif
// Board-local original40 / full6 serial DFlash execution smoke.
// No network, model compilation, speculative state, or numerical acceptance.
#include "qwen35_dflash/sha256.hpp"
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
#include "qwen35_dflash/resident_session.hpp"
#endif
#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/mman.h>
#include <signal.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using qwen35::dflash::Sha256;
using qwen35::dflash::Sha256File;
namespace {
#ifdef QWEN35_BOARD_SERIAL_HOST_FAKE
constexpr const char* kEvidence = "host-fake-hiai";
#else
constexpr const char* kEvidence = "device-hiai";
#endif
constexpr std::size_t kVocab=248320, kWidth=16384, kFeatureBytes=32768;
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
#if defined(QWEN35_RESIDENT_MIXED)
constexpr const char* kResidentMode="resident-mixed";
constexpr const char* kTargetLifecyclePolicy="v2-target-v1-draft-resident-v1";
constexpr const char* kResidentLoadApi="target-v2-draft-v1";
#elif defined(QWEN35_RESIDENT_V1)
constexpr const char* kResidentMode="resident-v1";
constexpr const char* kTargetLifecyclePolicy="legacy-v1-file-target-buffer-draft-v1";
constexpr const char* kResidentLoadApi="legacy-v1-mixed";
#else
constexpr const char* kResidentMode="resident-v2";
constexpr const char* kTargetLifecyclePolicy="init-before-weights-close-weights-first-v1";
constexpr const char* kResidentLoadApi="file-v2";
#endif
#endif
using Bytes=std::vector<std::uint8_t>;
using Clock=std::chrono::steady_clock;
std::uint64_t ElapsedMs(Clock::time_point start) {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now()-start).count());
}
void Lifecycle(const std::string& phase,const std::string& detail) {
  std::cout<<"board-serial: "<<phase<<' '<<detail<<'\n'<<std::flush;
}
[[noreturn]] void Fail(const std::string& text) { throw std::runtime_error(text); }
void Require(bool ok,const std::string& text) { if(!ok) Fail(text); }
std::vector<std::string> Split(const std::string& s,char delim) {
  std::vector<std::string> v; std::size_t begin=0;
  for(;;) { auto end=s.find(delim,begin); v.push_back(s.substr(begin,end-begin));
    if(end==std::string::npos) break;
    begin=end+1; } return v;
}
std::uint64_t Integer(const std::string& s,std::uint64_t maximum,bool zero=true) {
  Require(!s.empty() && s.size()<=20 && (s=="0" || s[0]!='0') &&
    std::all_of(s.begin(),s.end(),[](char c){return c>='0'&&c<='9';}),"invalid unsigned integer: "+s);
  std::uint64_t value=0; for(char c:s) { Require(value<=(UINT64_MAX-(c-'0'))/10,"integer overflow"); value=value*10+c-'0'; }
  Require(value<=maximum && (zero||value!=0),"integer outside bound: "+s); return value;
}
void HashSyntax(const std::string& s) { Require(std::regex_match(s,std::regex("[0-9a-f]{64}")),"invalid SHA256"); }
void Name(const std::string& s) { Require(s.size()<=256 && std::regex_match(s,std::regex("[A-Za-z_][A-Za-z0-9_.-]*")),"unsafe tensor name"); }
fs::path Absolute(const std::string& s) {
  Require(s.size()<=2048 && std::regex_match(s,std::regex("/[A-Za-z0-9_./-]+")),"unsafe absolute path: "+s);
  fs::path p(s); Require(p.lexically_normal()==p,"noncanonical path: "+s); return p;
}
void Safe(const fs::path& p,bool directory=false) {
  Require(p.is_absolute(),"path must be absolute"); fs::path current="/";
  for(const auto& part:p.relative_path()) { current/=part; struct stat s{};
    Require(::lstat(current.c_str(),&s)==0,"missing path: "+current.string());
    Require(!S_ISLNK(s.st_mode),"symlink rejected: "+current.string());
    if(current!=p || directory) Require(S_ISDIR(s.st_mode),"parent is not a directory: "+current.string());
    else Require(S_ISREG(s.st_mode),"not regular file: "+p.string());
  }
}
struct Stamp {
  std::uint64_t dev,ino,size,mode,uid,gid,links; std::int64_t mt,ct;
  bool operator==(const Stamp& b) const { return dev==b.dev&&ino==b.ino&&size==b.size&&mode==b.mode&&uid==b.uid&&gid==b.gid&&links==b.links&&mt==b.mt&&ct==b.ct; }
};
Stamp FromStat(const struct stat& s) { return {static_cast<std::uint64_t>(s.st_dev),static_cast<std::uint64_t>(s.st_ino),static_cast<std::uint64_t>(s.st_size),static_cast<std::uint64_t>(s.st_mode),s.st_uid,s.st_gid,static_cast<std::uint64_t>(s.st_nlink),static_cast<std::int64_t>(s.st_mtim.tv_sec)*1000000000+s.st_mtim.tv_nsec,static_cast<std::int64_t>(s.st_ctim.tv_sec)*1000000000+s.st_ctim.tv_nsec}; }
Stamp Stat(const fs::path& p) { Safe(p); struct stat s{}; Require(::lstat(p.c_str(),&s)==0,"stat failed"); return FromStat(s); }
std::string Hash(const Bytes& b) { return Sha256(std::string_view(reinterpret_cast<const char*>(b.data()),b.size())); }
Bytes Read(const fs::path& p,std::uint64_t extent) {
  auto before=Stat(p); Require(before.size==extent && extent<=8ULL*1024*1024,"file extent differs or exceeds budget: "+p.string());
  int fd=::open(p.c_str(),O_RDONLY|O_NOFOLLOW|O_NONBLOCK); Require(fd>=0,"open failed: "+p.string());
  struct stat s{}; bool same=::fstat(fd,&s)==0 && FromStat(s)==before;
  if(!same) {::close(fd);Fail("file changed at open");} Bytes b(extent); std::size_t n=0;
  while(n<b.size()) { auto got=::read(fd,b.data()+n,b.size()-n); if(got<0&&errno==EINTR)continue;
    if(got<=0) {::close(fd);Fail("short read");} n+=static_cast<std::size_t>(got); }
  std::uint8_t extra{}; auto end=::read(fd,&extra,1); same=::fstat(fd,&s)==0&&FromStat(s)==before; ::close(fd);
  Require(end==0&&same&&Stat(p)==before,"file changed while reading: "+p.string()); return b;
}
std::string Text(const fs::path& p,std::size_t maximum=1<<20) {
  auto n=Stat(p).size; Require(n<=maximum,"text budget exceeded: "+p.string()); auto b=Read(p,n); return {b.begin(),b.end()};
}
void Write(const fs::path& p,const Bytes& b) {
  Safe(p.parent_path(),true); int fd=::open(p.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600); Require(fd>=0,"cannot create new file: "+p.string());
  std::size_t n=0; while(n<b.size()) {auto got=::write(fd,b.data()+n,b.size()-n); if(got<0&&errno==EINTR)continue;
    if(got<=0){::close(fd);Fail("file write failed");} n+=static_cast<std::size_t>(got);} Require(::close(fd)==0,"close failed");
}
void Write(const fs::path& p,const std::string& s) { Write(p,Bytes(s.begin(),s.end())); }
void NewDir(const fs::path& p) { Safe(p.parent_path(),true); Require(::mkdir(p.c_str(),0700)==0,"new directory unavailable: "+p.string()); }
// Bounded strict JSON used solely for the existing native draft receipt.
struct J {
  enum Kind {Null,Bool,Number,String,Array,Object} kind=Null; std::string s; bool b=false; std::vector<J> a; std::map<std::string,J> o;
  J()=default; J(const char* v):kind(String),s(v){} J(std::string v):kind(String),s(std::move(v)){}
  J(std::uint64_t v):kind(Number),s(std::to_string(v)){} J(int v):J(static_cast<std::uint64_t>(v)){} J(unsigned int v):J(static_cast<std::uint64_t>(v)){} J(bool v):kind(Bool),b(v){}
  static J Arr(){J j;j.kind=Array;return j;} static J Obj(){J j;j.kind=Object;return j;}
  J& operator[](const std::string& key) {if(kind==Null)kind=Object;Require(kind==Object,"not JSON object");return o[key];}
  const J& at(const std::string& key)const {Require(kind==Object&&o.count(key),"missing JSON key: "+key);return o.at(key);}
  std::string str()const {Require(kind==String,"JSON string required");return s;}
  std::uint64_t num(std::uint64_t max=UINT64_MAX)const {Require(kind==Number,"JSON integer required");return Integer(s,max);}
};
std::string Quote(const std::string& s) {std::ostringstream o;o<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')o<<'\\'<<c;else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else o<<c;}o<<'"';return o.str();}
std::string Dump(const J& j) {if(j.kind==J::Null)return "null";if(j.kind==J::Bool)return j.b?"true":"false";if(j.kind==J::String)return Quote(j.s);if(j.kind==J::Number)return j.s;
  std::string s=j.kind==J::Array?"[":"{";bool first=true;if(j.kind==J::Array){for(const auto& x:j.a){if(!first)s+=',';first=false;s+=Dump(x);}}
  else{for(const auto& x:j.o){if(!first)s+=',';first=false;s+=Quote(x.first)+":"+Dump(x.second);}}return s+(j.kind==J::Array?"]":"}");}
struct Parser {
  const std::string& s;std::size_t i=0,nodes=0; void ws(){while(i<s.size()&&(s[i]==' '||s[i]=='\n'||s[i]=='\t'||s[i]=='\r'))++i;}
  std::string string(){Require(i<s.size()&&s[i++]=='"',"JSON string missing");std::string out;while(i<s.size()){unsigned char c=s[i++];if(c=='"')return out;
    Require(c>=32,"JSON string control");if(c!='\\'){out+=c;continue;}Require(i<s.size(),"JSON escape");char e=s[i++];
    if(e=='"'||e=='\\'||e=='/')out+=e;else if(e=='n')out+='\n';else if(e=='r')out+='\r';else if(e=='t')out+='\t';else if(e=='b')out+='\b';else if(e=='f')out+='\f';
    else if(e=='u'){Require(i+4<=s.size(),"JSON unicode escape");unsigned v=0;for(int x=0;x<4;++x){char h=s[i++];Require(std::isxdigit(static_cast<unsigned char>(h)),"JSON unicode hex");v=v*16+(h<='9'?h-'0':(h|32)-'a'+10);}Require(v>=32&&v<127,"nonASCII JSON escape rejected");out+=char(v);}else Fail("JSON escape invalid");}Fail("unterminated JSON string");}
  J value(int depth=0){Require(depth<24&&++nodes<=30000,"JSON budget exceeded");ws();Require(i<s.size(),"JSON truncated");char c=s[i];J j;
    if(c=='"')return J(string());
    if(c=='{'||c=='['){++i;j=c=='{'?J::Obj():J::Arr();ws();if(i<s.size()&&s[i]==(c=='{'?'}':']')){++i;return j;}
      for(;;){if(c=='{'){ws();auto key=string();ws();Require(i<s.size()&&s[i++]==':',"JSON colon");Require(!j.o.count(key),"duplicate JSON key");j.o.emplace(key,value(depth+1));}else j.a.push_back(value(depth+1));
        ws();Require(i<s.size(),"JSON collection truncated");char end=s[i++];if(end==(c=='{'?'}':']'))return j;Require(end==',',"JSON comma");}}
    for(auto p:std::vector<std::pair<std::string,J>>{{"true",J(true)},{"false",J(false)},{"null",J()}}){if(s.compare(i,p.first.size(),p.first)==0){i+=p.first.size();return p.second;}}
    std::size_t begin=i;if(s[i]=='-')++i;Require(i<s.size(),"JSON number");if(s[i]=='0')++i;else{Require(s[i]>='1'&&s[i]<='9',"JSON value invalid");while(i<s.size()&&std::isdigit(static_cast<unsigned char>(s[i])))++i;}
    if(i<s.size()&&s[i]=='.'){++i;std::size_t at=i;while(i<s.size()&&std::isdigit(static_cast<unsigned char>(s[i])))++i;Require(i>at,"JSON fraction");}
    if(i<s.size()&&(s[i]=='e'||s[i]=='E')){++i;if(i<s.size()&&(s[i]=='+'||s[i]=='-'))++i;std::size_t at=i;while(i<s.size()&&std::isdigit(static_cast<unsigned char>(s[i])))++i;Require(i>at,"JSON exponent");}
    j.kind=J::Number;j.s=s.substr(begin,i-begin);Require(j.s.size()<64,"JSON number budget");return j;}
  J parse(){J j=value();ws();Require(i==s.size(),"JSON trailing data");return j;}
};
void Atomic(const fs::path& p,const J& j) {auto tmp=p;tmp+=".new";Write(tmp,Dump(j)+"\n");Require(::rename(tmp.c_str(),p.c_str())==0,"atomic report rename failed");}
bool Shape(const J& a,const std::vector<std::uint64_t>& shape){if(a.kind!=J::Array||a.a.size()!=shape.size())return false;for(std::size_t i=0;i<shape.size();++i)if(a.a[i].num()!=shape[i])return false;return true;}
J Array(const std::vector<std::uint32_t>& v){J a=J::Arr();for(auto n:v)a.a.emplace_back(static_cast<std::uint64_t>(n));return a;}
J Ratio(std::uint64_t numerator,std::uint64_t denominator){if(!denominator)return J();J value;value.kind=J::Number;std::ostringstream out;out<<std::setprecision(17)<<static_cast<double>(numerator)/static_cast<double>(denominator);value.s=out.str();return value;}
struct File {fs::path path;std::string hash;std::uint64_t bytes;Stamp stamp;bool size_only=false;};
File Observe(const fs::path& p,std::uint64_t expected,const std::string& hash="") {auto before=Stat(p);Require(before.size==expected,"file extent mismatch: "+p.string());
  auto actual=Sha256File(p);Require(Stat(p)==before,"file changed during hash: "+p.string());if(!hash.empty())Require(actual==hash,"file SHA256 mismatch: "+p.string());return {p,actual,expected,before};}
void Stable(const File& f) {if(f.size_only){Require(Stat(f.path).size==f.bytes,"file extent changed: "+f.path.string());return;}Require(Stat(f.path)==f.stamp,"pinned file changed: "+f.path.string());}
// One stable file read supplies both its SHA observation and consumed bytes.
// Keep the pre-read baseline through hashing; a later read cannot silently
// establish a new identity after a read/hash gap.
struct FileBytes { File file; Bytes bytes; };
FileBytes ReadObserved(const fs::path& p,std::uint64_t expected,const std::string& hash="") {
  auto before=Stat(p);Require(before.size==expected,"file extent mismatch: "+p.string());
  auto bytes=Read(p,expected);Require(Stat(p)==before,"file changed during observed read: "+p.string());
  auto actual=Hash(bytes);Require(Stat(p)==before,"file changed during observed hash: "+p.string());
  if(!hash.empty())Require(actual==hash,"file SHA256 mismatch: "+p.string());
  return {{p,actual,expected,before},std::move(bytes)};
}
J Record(const File& f,const fs::path& root){J j=J::Obj();j["path"]=fs::relative(f.path,root).generic_string();j["bytes"]=f.bytes;j["sha256"]=f.hash;return j;}
struct Tensor {std::uint32_t index;std::string name,type;std::uint64_t bytes;};
struct State {std::string ref;std::uint32_t in,out;std::uint64_t bytes;};
struct PrefillState {std::string decode_ref,ref;std::uint32_t in,out;std::uint64_t bytes;};
struct Config {std::map<std::string,File> files;std::map<std::string,std::string> strings;std::vector<Tensor> inputs,outputs,prefill_inputs,prefill_outputs;std::vector<State> states;std::vector<PrefillState> prefill_states;std::vector<std::uint32_t> prompt;std::uint32_t maxnew=0,k=0,minrounds=0,timeout=0;std::uint64_t startup_hash_ms=0;bool dump=true,prefill=false,fast=false,draft_preread=false,incremental=false,size_only=false;Clock::time_point started=Clock::now();File config;
#ifdef QWEN35_TEXT_TOKENIZER
  std::shared_ptr<qwen35::dflash::TextTokenizer> tokenizer;
  std::uint64_t text_setup_us=0;
#endif
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
  qwen35::dflash::IncrementalDraftAbi incremental_abi;
  std::vector<qwen35::dflash::HiaiBoundRuntimeTensor> resident_inputs,resident_outputs,prefill_bound_inputs,prefill_bound_outputs;
#endif
};
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
void ReadRuntimeAbi(Config& c,const std::string& key,const std::string& role,
    const std::vector<Tensor>& input_wire,const std::vector<Tensor>& output_wire,
    std::vector<qwen35::dflash::HiaiBoundRuntimeTensor>& bound_inputs,
    std::vector<qwen35::dflash::HiaiBoundRuntimeTensor>& bound_outputs) {
  const auto& file=c.files.at(key);auto text=Text(file.path,1<<20);Stable(file);
  auto lines=Split(text,'\n');Require(!lines.empty()&&lines.back().empty(),"resident ABI needs final LF");lines.pop_back();
  Require(!lines.empty()&&lines[0]=="QWEN35_RESIDENT_ABI\t1","resident ABI header differs");
  for(std::size_t n=1;n<lines.size();++n){auto fields=Split(lines[n],'\t');Require(fields.size()==8&&(fields[0]==role+"_input"||fields[0]==role+"_output"),"resident ABI row invalid");
    auto& rows=fields[0]==role+"_input"?bound_inputs:bound_outputs;
    const auto& declared=fields[0]==role+"_input"?input_wire:output_wire;
    auto index=Integer(fields[1],100);Require(index==rows.size()&&index<declared.size(),"resident ABI ordered index differs");
    qwen35::dflash::HiaiBoundRuntimeTensor bound;bound.descriptor.index=index;bound.descriptor.tensor_name=fields[2];bound.descriptor.tensor_name_status="OBSERVED";
    bound.dtype=fields[3];bound.descriptor.raw_dtype=Integer(fields[4],INT32_MAX);bound.descriptor.raw_format=Integer(fields[5],INT32_MAX);
    std::uint64_t elements=1;for(const auto& dim:Split(fields[6],',')){auto value=Integer(dim,INT32_MAX,false);Require(elements<=(8ULL<<20)/value,"resident ABI tensor extent exceeds budget");elements*=value;bound.descriptor.shape.push_back(value);}
    Require(bound.descriptor.shape.size()<=8&&!bound.descriptor.tensor_name.empty()&&bound.descriptor.tensor_name.size()<=256,"resident descriptor rank/name invalid");
    bound.bytes=Integer(fields[7],8ULL<<20,false);const std::map<std::string,std::pair<std::string,unsigned>> types={{"FLOAT16",{"FP16",2}},{"FLOAT32",{"FP32",4}},{"INT8",{"INT8",1}},{"INT16",{"INT16",2}},{"INT32",{"INT32",4}},{"INT64",{"INT64",8}}};
    auto type=types.find(bound.dtype);Require(type!=types.end()&&elements*type->second.second==bound.bytes&&declared[index].type==type->second.first&&declared[index].bytes==bound.bytes&&declared[index].name==bound.descriptor.tensor_name,"resident exact descriptor differs from target wire ABI");rows.push_back(std::move(bound));
  }
  Require(bound_inputs.size()==89&&bound_outputs.size()==82,"resident ABI requires exact89/82 descriptors");
}
#endif
Config Load(const fs::path& path){Config c;auto configsize=Stat(path).size;Require(configsize<=65536,"config exceeds64KiB");auto configdata=ReadObserved(path,configsize);c.config=configdata.file;std::string text(configdata.bytes.begin(),configdata.bytes.end());
  for(unsigned char x:text)Require((x>=32&&x<=126)||x=='\n'||x=='\t',"TSV must be ASCII LF");
  auto lines=Split(text,'\n');Require(lines.back().empty(),"config needs final LF");lines.pop_back();
  Require(!lines.empty()&&lines[0]=="QWEN35_BOARD_SERIAL\t1","config header mismatch");
  if(lines.size()>1 && lines[1]=="runtime_profile\tcompact-memory-v1") {
#ifdef QWEN35_RESIDENT_MIXED
    c.fast=true;
#else
    Fail("compact-memory-v1 requires the mixed resident binary");
#endif
  }
  c.size_only=std::find(lines.begin(),lines.end(),"file_validation\tsize-and-abi-only")!=lines.end();
  Require(!c.size_only||c.fast,"size-only files require compact resident runtime");c.config.size_only=c.size_only;
  std::set<std::string> filekeys={"target_model","prefill_model","draft_model","draft_abi","target_tool","draft_runner","weight_table","scale_table"};
  std::set<std::string> strkeys={"target_library_path","draft_library_path","runtime_version","ddk_id","firmware_id","prompt","max_new_tokens","max_proposals","min_draft_rounds","timeout_ms"};
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
  filekeys.insert("resident_abi");
#if defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
  filekeys.erase("target_tool");filekeys.erase("draft_runner");filekeys.insert("prefill_abi");
#endif
  strkeys.insert("execution_mode");strkeys.insert("target_lifecycle_policy");
#endif
  const bool text_mode=std::find(lines.begin(),lines.end(),"text_io\tqwen35-text-v1")!=lines.end();
  if(text_mode){
#ifdef QWEN35_TEXT_TOKENIZER
    filekeys.insert("tokenizer_bundle");filekeys.insert("prompt_text");strkeys.insert("text_io");strkeys.insert("prompt_format");
#else
    Fail("text I/O requires the current mixed resident binary");
#endif
  }
  for(std::size_t n=1;n<lines.size();++n){auto f=Split(lines[n],'\t');Require(!f[0].empty(),"empty config row");
    if(f[0]=="draft_variant") {
#ifdef QWEN35_RESIDENT_MIXED
      Require(f.size()==2&&f[1]=="incremental-context-kv-v1"&&!c.incremental,"invalid/duplicate Draft variant");c.incremental=true;continue;
#else
      Fail("incremental Draft requires resident-mixed binary");
#endif
    }
    if(f[0]=="draft_preread"){
#ifdef QWEN35_RESIDENT_MIXED
      Require(f.size()==2&&f[1]=="sequential-1m-v1"&&!c.draft_preread,"unsupported/duplicate draft preread");c.draft_preread=true;continue;
#else
      Fail("draft preread requires mixed residency");
#endif
    }
    if(f[0]=="runtime_profile"){Require(n==1&&c.fast,"runtime profile must be first and supported");continue;}
    if(f[0]=="file_validation"){Require(n==2&&c.size_only&&f.size()==2,"file validation must follow runtime profile");continue;}
    if(filekeys.count(f[0])){Require(f.size()==4&&!c.files.count(f[0]),"file row malformed/duplicate");HashSyntax(f[2]);auto bytes=Integer(f[3],32ULL<<30,false);auto p=Absolute(f[1]);
      if(f[0]=="weight_table")Require(bytes==kVocab*2048ULL,"weight table extent mismatch");
      else if(f[0]=="scale_table")Require(bytes==kVocab*4ULL,"scale table extent mismatch");
      else if(f[0]=="prompt_text")Require(bytes<=16384,"text prompt exceeds16KiB");
      else if(f[0]=="tokenizer_bundle")Require(bytes<=16ULL<<20,"tokenizer bundle exceeds16MiB");
      else if(f[0]=="draft_model")Require(bytes<=UINT32_MAX,"draft model exceeds legacy32-bit extent");
      else if(f[0]=="draft_abi"||f[0]=="resident_abi"||f[0]=="prefill_abi")Require(bytes<=1ULL<<20,"draft/resident ABI exceeds budget");
      else Require(bytes<=64ULL<<20,"model/tool exceeds budget");
      // Full table/model hashes occur once at initialization, never per token.
      Lifecycle(c.fast?"startup-bind-begin":"startup-hash-begin","role="+f[0]+" bytes="+std::to_string(bytes));
      auto hash_begin=Clock::now();
      const bool metadata=f[0]=="tokenizer_bundle"||f[0]=="prompt_text"||f[0]=="draft_abi"||f[0]=="resident_abi"||f[0]=="prefill_abi";
      if(c.size_only||(c.fast&&!metadata)){auto stamp=Stat(p);Require(stamp.size==bytes,"deployment payload extent differs");c.files.emplace(f[0],File{p,f[2],bytes,stamp,c.size_only});}
      else c.files.emplace(f[0],Observe(p,bytes,f[2]));
      auto hash_ms=ElapsedMs(hash_begin);c.startup_hash_ms+=hash_ms;
      Lifecycle(c.fast?"startup-bind-end":"startup-hash-end","role="+f[0]+" wall_ms="+std::to_string(hash_ms));}
    else if(f[0]=="dump"){Require(f.size()==2&&!c.strings.count("dump")&&(f[1]=="0"||f[1]=="1"),"dump must be one explicit0/1 singleton");c.dump=f[1]=="1";c.strings.emplace("dump",f[1]);}
    else if(strkeys.count(f[0])){Require(f.size()==2&&!f[1].empty()&&!c.strings.count(f[0]),"singleton row malformed/duplicate");c.strings.emplace(f[0],f[1]);}
    else if(f[0]=="target_input"||f[0]=="target_output"||f[0]=="prefill_input"||f[0]=="prefill_output"){Require(f.size()==5,"tensor row malformed");auto& v=f[0]=="target_input"?c.inputs:f[0]=="target_output"?c.outputs:f[0]=="prefill_input"?c.prefill_inputs:c.prefill_outputs;
      auto index=Integer(f[1],100);Require(index==v.size(),"tensor indices must be ordered/contiguous");if(f[0]=="target_input"||f[0]=="prefill_input")Name(f[2]);else Require(!f[2].empty()&&f[2].size()<=256,"runtime output name exceeds budget");Require(std::set<std::string>{"FP16","FP32","INT8","INT16","INT32","INT64"}.count(f[3]),"tooltype invalid");
      v.push_back({static_cast<std::uint32_t>(index),f[2],f[3],Integer(f[4],8ULL<<20,false)});}
    else if(f[0]=="state"){Require(f.size()==5,"state row malformed");Name(f[1]);c.states.push_back({f[1],static_cast<std::uint32_t>(Integer(f[2],88)),static_cast<std::uint32_t>(Integer(f[3],81)),Integer(f[4],8ULL<<20,false)});}
    else if(f[0]=="prefill_state"){Require(f.size()==6,"prefill state row malformed");Name(f[1]);Name(f[2]);c.prefill_states.push_back({f[1],f[2],static_cast<std::uint32_t>(Integer(f[3],88)),static_cast<std::uint32_t>(Integer(f[4],81)),Integer(f[5],8ULL<<20,false)});}
    else Fail("unknown config row: "+f[0]);}
  Require(!c.fast||(!c.dump&&c.strings.count("dump")),"compact memory requires explicit dump=0");
  c.prefill=c.files.count("prefill_model")!=0;
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
  Require(c.prefill&&c.strings.count("execution_mode")&&c.strings.at("execution_mode")==kResidentMode,"resident binary requires explicit "+std::string(kResidentMode)+" mode and one original prefill");
  Require(c.strings.count("target_lifecycle_policy")&&c.strings.at("target_lifecycle_policy")==kTargetLifecyclePolicy,"resident binary requires explicit current target lifecycle policy");
#endif
  Require(c.files.size()==filekeys.size()-1+static_cast<std::size_t>(c.prefill)&&c.strings.size()==strkeys.size()+c.strings.count("dump"),"config fields missing");
  Require(c.prefill?(c.prefill_inputs.size()==89&&c.prefill_outputs.size()==82&&c.prefill_states.size()==80):(c.prefill_inputs.empty()&&c.prefill_outputs.empty()&&c.prefill_states.empty()),"prefill config must be complete89/82/80 or entirely absent");
  Require(c.files.at("weight_table").bytes==kVocab*2048ULL&&c.files.at("scale_table").bytes==kVocab*4ULL,"original table extents mismatch");
  Require(c.files.at("target_model").bytes<=64ULL<<20&&c.files.at("draft_model").bytes<=UINT32_MAX&&c.files.at("draft_abi").bytes<=1<<20,"model/ABI budget mismatch");
  for(auto key:{"target_library_path","draft_library_path"})for(const auto& s:Split(c.strings.at(key),':'))Safe(Absolute(s),true);
#if !defined(QWEN35_RESIDENT_V1) && !defined(QWEN35_RESIDENT_MIXED)
  for(auto key:{"target_tool","draft_runner"})Require(::access(c.files.at(key).path.c_str(),X_OK)==0,"child executable is not executable");
#endif
  for(const auto& s:Split(c.strings.at("prompt"),','))c.prompt.push_back(static_cast<std::uint32_t>(Integer(s,kVocab-1)));
#ifdef QWEN35_TEXT_TOKENIZER
  if(text_mode){
    auto start=Clock::now();const auto& file=c.files.at("prompt_text");auto bytes=Read(file.path,file.bytes);Stable(file);
    c.tokenizer=std::make_shared<qwen35::dflash::TextTokenizer>(c.files.at("tokenizer_bundle").path.string());
    auto encoded=c.tokenizer->Encode(std::string(bytes.begin(),bytes.end()),c.strings.at("prompt_format"));
    Require(encoded==c.prompt,"board tokenizer differs from original tokenizer admission IDs; no model launched");
    c.prompt=std::move(encoded);c.text_setup_us=std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-start).count();
    Lifecycle("text-encoded","tokens="+std::to_string(c.prompt.size())+" wall_us="+std::to_string(c.text_setup_us));
  }
#endif
  c.maxnew=Integer(c.strings.at("max_new_tokens"),63,false);c.k=Integer(c.strings.at("max_proposals"),15,false);c.minrounds=Integer(c.strings.at("min_draft_rounds"),63,false);Require(c.minrounds>=2&&c.minrounds<=c.maxnew-1,"minimum draft rounds must be at least2 and no more than generation budget minus anchor");c.timeout=Integer(c.strings.at("timeout_ms"),7200000,false);
  Require(!c.prompt.empty()&&c.prompt.size()<=63&&c.prompt.size()+c.maxnew<=64&&c.inputs.size()==89&&c.outputs.size()==82&&c.states.size()==80,"requires complete bounded S64/89input/82output/80state contract");
  std::map<std::string,std::pair<std::string,std::uint64_t>> controls={{"arg1_1",{"INT8",2048}},{"arg113_1",{"FP32",8192}},{"arg107_1",{"INT32",4}},{"arg2_1",{"INT32",4}},{"arg1238_1",{"INT32",4}},{"arg0_1",{"FP32",4}},{"arg114_1",{"INT64",8}},{"arg14_1",{"INT32",512}},{"arg18_1",{"INT16",2}}};
  std::set<std::string> names,refs;std::set<unsigned> ins,outs;std::uint64_t total=0;
  for(const auto& t:c.inputs){Require(names.insert(t.name).second,"duplicate input name");total+=t.bytes;auto it=controls.find(t.name);if(it!=controls.end()){Require(t.type==it->second.first&&t.bytes==it->second.second,"control ABI mismatch");controls.erase(it);}}
  Require(controls.empty()&&total<=128ULL<<20,"missing control or excessive input bytes");
  for(const auto& s:c.states){Require(refs.insert(s.ref).second&&ins.insert(s.in).second&&outs.insert(s.out).second,"duplicate state owner/port");
    Require(std::regex_match(s.ref,std::regex("RefData_([0-9]|[1-7][0-9])"))&&c.inputs[s.in].name==s.ref&&c.inputs[s.in].type=="FP16"&&c.outputs[s.out].type=="FP16"&&s.bytes==c.inputs[s.in].bytes&&s.bytes==c.outputs[s.out].bytes,"state mapping differs from typed ABI");
    Require(s.out!=0&&s.out!=81,"state cannot bind logits/features");}
  Require(c.outputs[0].type=="FP32"&&c.outputs[0].bytes==kVocab*4&&c.outputs[81].type=="FP16"&&c.outputs[81].bytes==kFeatureBytes,"full logits/features ABI mismatch");
  names.clear();for(const auto& t:c.outputs)Require(names.insert(t.name).second,"duplicate runtime output name");
  if(c.prefill){
    auto controls64=std::map<std::string,std::pair<std::string,std::uint64_t>>{{"arg1_1",{"INT8",64*2048}},{"arg113_1",{"FP32",64*8192}},{"arg107_1",{"INT32",64*4}},{"arg2_1",{"INT32",64*4}},{"arg1238_1",{"INT32",4}},{"arg0_1",{"FP32",64*4}},{"arg114_1",{"INT64",8}},{"arg14_1",{"INT32",512}},{"arg18_1",{"INT16",2}}};
    names.clear();total=0;for(const auto& t:c.prefill_inputs){Require(names.insert(t.name).second,"duplicate prefill input name");total+=t.bytes;auto it=controls64.find(t.name);if(it!=controls64.end()){Require(t.type==it->second.first&&t.bytes==it->second.second,"prefill64 control ABI mismatch");controls64.erase(it);}}
    Require(controls64.empty()&&total<=128ULL<<20,"missing prefill64 control or excessive input extent");
    std::map<std::string,State> decode;for(const auto& s:c.states)decode.emplace(s.ref,s);
    std::set<std::string> decode_refs,prefill_refs;ins.clear();outs.clear();
    for(const auto& s:c.prefill_states){
      Require(decode.count(s.decode_ref)&&decode_refs.insert(s.decode_ref).second&&prefill_refs.insert(s.ref).second&&ins.insert(s.in).second&&outs.insert(s.out).second,"prefill/decode state mapping is not a complete bijection");
      Require(std::regex_match(s.ref,std::regex("RefData_([0-9]|[1-7][0-9])"))&&c.prefill_inputs[s.in].name==s.ref&&c.prefill_inputs[s.in].type=="FP16"&&c.prefill_outputs[s.out].type=="FP16"&&s.out!=0&&s.out!=81&&s.bytes==c.prefill_inputs[s.in].bytes&&s.bytes==c.prefill_outputs[s.out].bytes&&s.bytes==decode.at(s.decode_ref).bytes,"prefill/decode state typed extent or name differs");
    }
    Require(c.prefill_outputs[0].type=="FP32"&&c.prefill_outputs[0].bytes==kVocab*4&&c.prefill_outputs[81].type=="FP16"&&c.prefill_outputs[81].bytes==64*kFeatureBytes,"prefill full logits/features ABI mismatch");
    names.clear();for(const auto& t:c.prefill_outputs)Require(names.insert(t.name).second,"duplicate prefill runtime output name");
  }
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
  ReadRuntimeAbi(c,"resident_abi","target",c.inputs,c.outputs,c.resident_inputs,c.resident_outputs);
#if defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
  ReadRuntimeAbi(c,"prefill_abi","prefill",c.prefill_inputs,c.prefill_outputs,c.prefill_bound_inputs,c.prefill_bound_outputs);
#endif
#endif
#ifdef QWEN35_RESIDENT_MIXED
  if(c.incremental) {
    Require(c.fast&&!c.dump,"incremental Draft requires compact-memory mode");
    const auto raw=Text(c.files.at("draft_abi").path);const auto abi=Parser{raw}.parse();
    Require(abi.at("schema_version").num()==1&&abi.at("variant").str()=="incremental-context-kv-v1"&&
      abi.at("model_sha256").str()==c.files.at("draft_model").hash,"incremental Draft ABI/model binding differs");
    auto& spec=c.incremental_abi;
    spec.context_rows=abi.at("context_rows").num(64);spec.capacity=abi.at("cache_capacity").num(4096);spec.candidates=abi.at("max_draft_tokens").num(15);
    auto parse=[](const J& rows,auto& dest){
      Require(rows.kind==J::Array&&rows.a.size()<=17,"incremental ABI rows invalid");
      for(const auto& row:rows.a){Require(row.at("index").num()==dest.size(),"incremental ABI slot index differs");
        qwen35::dflash::HiaiTensorSpec tensor;tensor.name=row.at("name").str();tensor.dtype=row.at("dtype").str();
        const auto& shape=row.at("shape");Require(shape.kind==J::Array&&shape.a.size()<=4,"incremental shape invalid");
        for(const auto& dim:shape.a)tensor.shape.push_back(dim.num(262144));
        Require(qwen35::dflash::HiaiTensorBytes(tensor)==row.at("bytes").num(),"incremental ABI bytes differ");dest.push_back(tensor);
      }
    };
    parse(abi.at("inputs"),spec.inputs);parse(abi.at("outputs"),spec.outputs);spec.Validate();
    Require(c.k<=spec.candidates&&c.prompt.size()+c.maxnew-1<=spec.capacity,"incremental Draft gear/cache cannot cover request");
  }
#endif
  for(const auto& x:c.files)Stable(x.second);
  Stable(c.config);return c;
}
Bytes Scalar(std::uint64_t value,std::size_t bytes){Bytes b(bytes);for(std::size_t i=0;i<bytes;++i)b[i]=(value>>(8*i))&255;return b;}
void FiniteHalf(const Bytes& b){Require(b.size()%2==0,"FP16 extent");for(std::size_t i=0;i<b.size();i+=2)Require(((b[i]|(b[i+1]<<8))&0x7c00)!=0x7c00,"nonfinite FP16 output");}
std::uint32_t Argmax(const Bytes& b){Require(b.size()==kVocab*4,"logits extent");float best=0;std::uint32_t index=0;for(std::uint32_t i=0;i<kVocab;++i){std::uint32_t bits=0;for(int n=0;n<4;++n)bits|=std::uint32_t(b[4*i+n])<<(8*n);float v;std::memcpy(&v,&bits,4);Require(std::isfinite(v),"nonfinite target logits");if(i==0||v>best){best=v;index=i;}}return index;}
Bytes TableRow(const File& f,std::uint32_t token,std::size_t width){Stable(f);int fd=::open(f.path.c_str(),O_RDONLY|O_NOFOLLOW|O_NONBLOCK);Require(fd>=0,"table open failed");struct stat s{};bool okay=::fstat(fd,&s)==0&&FromStat(s)==f.stamp;Bytes b(width);std::size_t done=0;
  while(okay&&done<width){auto got=::pread(fd,b.data()+done,width-done,static_cast<off_t>(token)*width+done);if(got<0&&errno==EINTR)continue;if(got<=0){okay=false;break;}done+=got;}okay=okay&&::fstat(fd,&s)==0&&FromStat(s)==f.stamp;::close(fd);Require(okay,"table identity/read failed");Stable(f);return b;}
struct Child {std::uint64_t pid;int exitcode;std::uint64_t wall_ms;};
Child Run(const std::vector<std::string>& args,const fs::path& cwd,const std::string& libraries,const fs::path& logs,J& progress,J& children,const fs::path& out){
  J argv=J::Arr();for(const auto& a:args)argv.a.emplace_back(a);J cmd=J::Obj();cmd["argv"]=argv;cmd["cwd"]=cwd.string();cmd["LD_LIBRARY_PATH"]=libraries;Write(logs/"command.json",Dump(cmd)+"\n");
  progress["last_child_pid"]=J();progress["last_child_exitcode"]=J();progress["child_status"]="STARTING";Atomic(out/"progress.json",progress);
  auto child_begin=Clock::now();
  pid_t pid=::fork();Require(pid>=0,"fork failed");if(pid==0){int a=::open((logs/"tool.stdout").c_str(),O_WRONLY|O_CREAT|O_EXCL,0600),b=::open((logs/"tool.stderr").c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    if(a<0||b<0||::dup2(a,STDOUT_FILENO)<0||::dup2(b,STDERR_FILENO)<0||::chdir(cwd.c_str())!=0||::setenv("LD_LIBRARY_PATH",libraries.c_str(),1)!=0)_exit(126);
    ::close(a);::close(b);std::vector<char*> v;for(const auto& s:args)v.push_back(const_cast<char*>(s.c_str()));v.push_back(nullptr);::execv(v[0],v.data());_exit(127);}
  progress["last_child_pid"]=static_cast<std::uint64_t>(pid);progress["last_child_exitcode"]=J();progress["child_status"]="RUNNING";
  J child=J::Obj();child["pid"]=static_cast<std::uint64_t>(pid);child["exitcode"]=J();children.a.push_back(child);
  progress["launched_child_count"]=children.a.size();progress["active_child_count"]=1;
  Lifecycle("child-start","pid="+std::to_string(pid)+" executable="+args[0]);
  Write(logs/"tool.pid",std::to_string(pid)+"\n");Atomic(out/"progress.json",progress);
  int status=0;pid_t waited;do{waited=::waitpid(pid,&status,0);}while(waited<0&&errno==EINTR);Require(waited==pid,"waitpid failed; completion unknown");int rc=WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status);
  auto wall_ms=ElapsedMs(child_begin);Lifecycle("child-end","pid="+std::to_string(pid)+" exitcode="+std::to_string(rc)+" wall_ms="+std::to_string(wall_ms));
  Write(logs/"tool.exitcode",std::to_string(rc)+"\n");progress["last_child_exitcode"]=rc;progress["child_status"]="EXITED";children.a.back()["exitcode"]=rc;progress["completed_child_count"]=children.a.size();progress["active_child_count"]=0;Atomic(out/"progress.json",progress);return {static_cast<std::uint64_t>(pid),rc,wall_ms};
}
std::string Numbered(const std::string& prefix,std::size_t n){std::ostringstream s;s<<prefix<<std::setfill('0')<<std::setw(4)<<n;return s.str();}
class MappedTable {
  void* data=MAP_FAILED;std::size_t bytes=0;
 public:
  explicit MappedTable(const File& f):bytes(f.bytes){
    int fd=::open(f.path.c_str(),O_RDONLY|O_NOFOLLOW);Require(fd>=0,"embedding table open failed");
    struct stat st{};if(::fstat(fd,&st)!=0||(f.size_only?static_cast<std::uint64_t>(st.st_size)!=f.bytes:!(FromStat(st)==f.stamp))){::close(fd);Fail("embedding table differs before mapping");}
    data=::mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);::close(fd);Require(data!=MAP_FAILED,"embedding table mmap failed");
  }
  ~MappedTable(){if(data!=MAP_FAILED)::munmap(data,bytes);}
  MappedTable(const MappedTable&)=delete;
  Bytes Row(std::uint32_t token,std::size_t width)const{
    Require(token<kVocab&&width&&std::uint64_t(token)*width+width<=bytes,"embedding row outside table");
    const auto* begin=static_cast<const std::uint8_t*>(data)+std::size_t(token)*width;return Bytes(begin,begin+width);
  }
};
class Supervisor {
  std::unique_ptr<MappedTable> weight_mapping,scale_mapping;
  std::vector<Bytes> memory_states; // indexed by the pinned decode runtime input slot
  std::uint64_t prefill_api_us=0;
  Bytes EmbeddingRow(bool scale,std::uint32_t token){
    if(c.fast)return (scale?scale_mapping:weight_mapping)->Row(token,scale?4:2048);
    return TableRow(c.files.at(scale?"scale_table":"weight_table"),token,scale?4:2048);
  }
  Config c;fs::path out,last_target_step;J loop=J::Obj(),progress=J::Obj(),children=J::Arr();std::vector<File> sources,inputs,outputs,states;std::vector<std::uint32_t> prefix,generated;Bytes features;std::uint32_t next=0;std::set<std::string> retrieve;std::map<std::string,File> kept;std::size_t targets=0,drafts=0,prefills=0,prompt_calls=0,round_count=0,considered=0,accepted=0,corrections=0,bonuses=0;std::uint64_t target_child_ms=0,draft_child_ms=0,prompt_child_ms=0,prefill_child_ms=0;std::map<std::uint64_t,std::string> zero_hashes;std::string stage="initialize";
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
  qwen35::dflash::ResidentSession session{kResidentLoadApi};
  std::unique_ptr<qwen35::dflash::IncrementalDraftCache> draft_cache;
  std::string session_id;
  std::uint64_t resident_call_index=0;
  J resident_memory=J::Arr();
#if defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
#ifdef QWEN35_RESIDENT_MIXED
  std::unique_ptr<qwen35::dflash::HiaiOriginalTargetExecutor> prefill_executor;
  qwen35::dflash::HiaiOriginalTargetLifecycle prefill_lifecycle;
#else
  std::unique_ptr<qwen35::dflash::HiaiOriginalTargetV1Executor> prefill_executor;
  qwen35::dflash::HiaiLegacyLifecycle prefill_lifecycle;
#endif
  std::uint64_t prefill_load_ms=0,prefill_process_ms=0,prefill_unload_ms=0,prefill_total_ms=0;
  J LegacyLifecycle(const qwen35::dflash::HiaiLegacyLifecycle& e) const {
    J value=J::Obj();value["model_load_api"]=e.model_load_api;value["lifecycle_policy"]=e.lifecycle_policy;value["model_name"]=e.model_name;
    value["runtime_tensor_names"]=e.runtime_tensor_names;value["runtime_dtypes"]=e.runtime_dtypes;value["external_weight_resolution"]=e.external_weight_resolution;
    value["client_init_attempted"]=e.client_init_attempted;value["client_initialized"]=e.client_initialized;value["load_attempted"]=e.load_attempted;value["loaded"]=e.loaded;value["unload_called"]=e.unload_called;value["closed"]=e.closed;
    auto status=[](const std::optional<std::int64_t>& v){J j;if(v){j.kind=J::Number;j.s=std::to_string(*v);}return j;};
    value["client_init_status"]=status(e.client_init_status);value["load_status"]=status(e.load_status);value["unload_status"]=status(e.unload_status);value["process_attempts"]=e.process_attempts;value["successful_processes"]=e.successful_processes;value["failure_stage"]=e.failure_stage;value["cleanup_error"]=e.cleanup_error;
    if(!e.load_phases.empty()){
      J profile=J::Obj();profile["scope"]="LOAD_PHASE_WALL_AND_PROCESS_COUNTER_DELTAS_NOT_NPU_OR_SYSTEM_IO";profile["phases"]=J::Arr();
      if(e.preread_requested){profile["preread_policy"]="sequential-1m-v1";profile["preread_bytes"]=e.preread_bytes;profile["preread_scratch_limit_bytes"]=1048576;}
      auto counter=[](const std::optional<std::uint64_t>& v){return v?J(*v):J();};
      for(const auto& p:e.load_phases){J row=J::Obj();row["phase"]=p.phase;row["wall_us"]=p.wall_us;row["completed"]=p.completed;
        row["minor_faults"]=counter(p.minor_faults);row["major_faults"]=counter(p.major_faults);row["read_bytes"]=counter(p.read_bytes);row["read_chars"]=counter(p.read_chars);profile["phases"].a.push_back(row);}
      value["load_profile"]=profile;
    }
    return value;
  }
  J LegacyObservedAbi(const qwen35::dflash::HiaiLegacyLifecycle& e) const {
    J abi=J::Obj();auto rows=[](const auto& source){J result=J::Arr();for(const auto& t:source){J row=J::Obj();row["index"]=static_cast<std::uint64_t>(t.index);row["shape"]=J::Arr();for(auto dim:t.shape)row["shape"].a.emplace_back(static_cast<std::uint64_t>(dim));row["raw_format"]=static_cast<std::uint64_t>(t.raw_format);row["bytes"]=static_cast<std::uint64_t>(t.bytes);result.a.push_back(row);}return result;};
    abi["inputs"]=rows(e.observed_inputs);abi["outputs"]=rows(e.observed_outputs);return abi;
  }
  void LegacyAbiEvidence(J& receipt,const qwen35::dflash::HiaiLegacyLifecycle& life) const {
    receipt["runtime_tensor_names"]=life.runtime_tensor_names;receipt["runtime_dtypes"]=life.runtime_dtypes;receipt["runtime_abi_observed_on_load"]=LegacyObservedAbi(life);
  }
  void ClosePrefill() {
    if(!prefill_executor)return;
    auto start=Clock::now();try{prefill_executor->Close();}catch(...){prefill_unload_ms+=ElapsedMs(start);prefill_lifecycle=prefill_executor->lifecycle();prefill_executor.reset();throw;}
    prefill_unload_ms+=ElapsedMs(start);prefill_lifecycle=prefill_executor->lifecycle();prefill_executor.reset();
  }
#endif
  J V2Lifecycle(const qwen35::dflash::HiaiOriginalTargetLifecycle& t) const {
    J target=J::Obj();target["weight_init_policy"]=t.weight_init_policy;target["weight_cleanup_policy"]=t.weight_cleanup_policy;target["ordering_validation"]=t.ordering_validation;
    target["weights_init_attempted"]=t.weights_init_attempted;target["weights_initialized"]=t.weights_initialized;target["manager_init_attempted"]=t.manager_init_attempted;target["manager_initialized"]=t.manager_initialized;
    auto status_value=[](const std::optional<std::int64_t>& status){J value;if(status){value.kind=J::Number;value.s=std::to_string(*status);}return value;};
    target["weights_init_status"]=status_value(t.weights_init_status);target["manager_init_status"]=status_value(t.manager_init_status);target["failure_stage"]=t.failure_stage;target["cleanup_error"]=t.cleanup_error;
    target["manager_deinit_called"]=t.manager_deinit_called;target["manager_deinit_returned"]=t.manager_deinit_returned;target["weights_deinit_called"]=t.weights_deinit_called;target["closed"]=t.closed;
    J deinit_status;deinit_status.kind=J::Number;deinit_status.s=std::to_string(t.weights_deinit_status);target["weights_deinit_status"]=deinit_status;target["run_attempts"]=t.run_attempts;target["successful_runs"]=t.successful_runs;
#ifdef QWEN35_RESIDENT_MIXED
    target["external_weight_release"]="NOT_INDEPENDENTLY_VALIDATED_MANAGER_DEINIT_ONLY";
#endif
    return target;
  }
  void MemorySnapshot(const std::string& phase) {
    J value=J::Obj();value["phase"]=phase;value["status"]="UNAVAILABLE";
    auto sample=[&](const char* path,const std::map<std::string,std::string>& fields){std::ifstream file(path);std::string line;std::size_t total=0;while(std::getline(file,line)&&total<65536){total+=line.size()+1;std::istringstream input(line);std::string name,unit;std::uint64_t number=0;if(input>>name>>number>>unit){auto selected=fields.find(name);if(selected!=fields.end()&&unit=="kB"){value[selected->second]=number;value["status"]="OBSERVED";}}}};
    sample("/proc/self/status",{{"VmRSS:","vm_rss_kib"},{"VmHWM:","vm_hwm_kib"}});sample("/proc/meminfo",{{"MemAvailable:","mem_available_kib"}});resident_memory.a.push_back(value);
  }
  void ResidentIdentity(J& value,std::uint64_t index) const {
    value["execution_mode"]=kResidentMode;value["session_id"]=session_id;
    value["process_id"]=static_cast<std::uint64_t>(::getpid());value["call_index"]=index;
  }
  J ResidentEvidence() const {
    const auto& e=session.evidence();J value=J::Obj();value["session_id"]=session_id;value["process_id"]=static_cast<std::uint64_t>(::getpid());
    value["stage"]=e.stage;value["status"]=e.status;value["model_load_api"]=e.model_load_api;
    value["target_model_name"]=e.target_model_name;value["draft_model_name"]=e.draft_model_name;
#ifdef QWEN35_RESIDENT_V1
    value["target_model_name_scope"]="V1_MODEL_DESCRIPTION_LABEL_NOT_RUNTIME_TENSOR_NAME";value["draft_model_name_scope"]="V1_MODEL_DESCRIPTION_LABEL_NOT_RUNTIME_TENSOR_NAME";
#else
    value["target_model_name_scope"]=e.target_loads?"SDK_SETNAME_GETNAME_VALIDATED_BEFORE_INIT":"REQUESTED_NOT_YET_LOADED";
#ifdef QWEN35_RESIDENT_MIXED
    value["draft_model_name_scope"]="V1_MODEL_DESCRIPTION_LABEL_NOT_RUNTIME_TENSOR_NAME";
#else
    value["draft_model_name_scope"]=e.draft_loads?"SDK_SETNAME_GETNAME_VALIDATED_BEFORE_INIT":"REQUESTED_NOT_YET_LOADED";
#endif
#endif
    value["cleanup_error"]=e.cleanup_error;value["target_load_attempts"]=e.target_load_attempts;value["draft_load_attempts"]=e.draft_load_attempts;
    value["target_loads"]=e.target_loads;value["draft_loads"]=e.draft_loads;value["target_run_attempts"]=e.target_run_attempts;value["target_runs"]=e.target_runs;
    value["draft_run_attempts"]=e.draft_run_attempts;value["draft_runs"]=e.draft_runs;value["target_close_calls"]=e.target_close_calls;value["draft_close_calls"]=e.draft_close_calls;
    value["target_load_ms"]=e.target_load_ms;value["draft_load_ms"]=e.draft_load_ms;value["target_run_ms"]=e.target_run_ms;value["draft_run_ms"]=e.draft_run_ms;value["close_ms"]=e.close_ms;
    value["closed"]=e.closed;value["poisoned"]=e.poisoned;
#ifdef QWEN35_RESIDENT_V1
    value["target_v1_lifecycle"]=LegacyLifecycle(e.target_v1_lifecycle);value["draft_v1_lifecycle"]=LegacyLifecycle(e.draft_v1_lifecycle);return value;
#else
    value["target_lifecycle"]=V2Lifecycle(e.target_lifecycle);
#ifdef QWEN35_RESIDENT_MIXED
    value["draft_v1_lifecycle"]=LegacyLifecycle(e.draft_v1_lifecycle);
#endif
    return value;
#endif
  }
  void ResidentCounts() {
    const auto& e=session.evidence();loop["schema_version"]=3;loop["dump"]=c.dump;loop["execution_mode"]=kResidentMode;loop["resident_lifecycle"]=ResidentEvidence();
    loop["target_lifecycle_policy"]=c.strings.at("target_lifecycle_policy");
    loop["resident_memory"]=J::Obj();loop["resident_memory"]["scope"]="PROCESS_RSS_NOT_NPU_ALLOCATED_MEMORY";loop["resident_memory"]["snapshots"]=resident_memory;
    loop["resident_abi_sha256"]=c.files.at("resident_abi").hash;
#ifdef QWEN35_RESIDENT_MIXED
    loop["prefill_abi_sha256"]=c.files.at("prefill_abi").hash;loop["prefill_lifecycle"]=V2Lifecycle(prefill_lifecycle);loop["model_residency"]="PERSISTENT_V2_TARGET_AND_V1_DRAFT_AFTER_V2_PREFILL_CLOSE";
    loop["model_load_api"]="target-v2-draft-v1";loop["model_load_apis"]=J::Obj();loop["model_load_apis"]["prefill"]="file-v2";loop["model_load_apis"]["target"]="file-v2";loop["model_load_apis"]["draft"]="legacy-buffer";
    loop["external_weight_release"]="NOT_INDEPENDENTLY_VALIDATED_MANAGER_DEINIT_ONLY";
#elif defined(QWEN35_RESIDENT_V1)
    loop["prefill_abi_sha256"]=c.files.at("prefill_abi").hash;loop["prefill_lifecycle"]=LegacyLifecycle(prefill_lifecycle);loop["model_residency"]="TWO_PERSISTENT_V1_CLIENTS_AFTER_PREFILL_CLIENT_UNLOAD";
    loop["model_load_api"]="legacy-v1-mixed";loop["model_load_apis"]=J::Obj();loop["model_load_apis"]["prefill"]="legacy-file";loop["model_load_apis"]["target"]="legacy-file";loop["model_load_apis"]["draft"]="legacy-buffer";
#else
    loop["model_residency"]="TWO_PERSISTENT_FILE_V2_MANAGERS_AFTER_PREFILL";
#endif
    auto& timing=loop["supervisor_timing"];timing["target_child_process_wall_ms"]=0;timing["draft_child_process_wall_ms"]=0;timing["target_child_processes_timed"]=0;timing["draft_child_processes_timed"]=0;
    timing["resident_target_load_ms"]=e.target_load_ms;timing["resident_draft_load_ms"]=e.draft_load_ms;timing["resident_target_run_ms"]=e.target_run_ms;timing["resident_draft_run_ms"]=e.draft_run_ms;timing["resident_close_ms"]=e.close_ms;
    auto phase=[](std::uint64_t calls,std::uint64_t ms){J p=J::Obj();p["calls"]=calls;p["total_ms"]=ms;p["mean_ms"]=Ratio(ms,calls);return p;};
    auto& phases=loop["phase_latency"];phases["timing_scope"]="RESIDENT_RUN_WALL_EXCLUDES_MODEL_LOAD_AND_CLOSE";phases["decode"]=phase(e.target_runs,e.target_run_ms);phases["draft"]=phase(e.draft_runs,e.draft_run_ms);
#ifdef QWEN35_RESIDENT_MIXED
    timing["prefill_child_process_wall_ms"]=0;timing["prefill_child_processes_timed"]=0;
    timing["prefill_model_load_ms"]=prefill_load_ms;timing["prefill_process_ms"]=prefill_process_ms;timing["prefill_model_unload_ms"]=prefill_unload_ms;timing["prefill_v2_total_ms"]=prefill_total_ms;
    phases["prefill"]=phase(prefills,prefill_total_ms);phases["prefill"]["native_api"]="Run";phases["prefill"]["timing_scope"]="PREFILL_V2_LOAD_RUN_CLOSE_WALL";
    phases["prompt_ingest"]["calls"]=prefills;phases["prompt_ingest"]["total_ms"]=prefill_total_ms;phases["prompt_ingest"]["mean_ms"]=Ratio(prefill_total_ms,prefills);phases["prompt_ingest"]["timing_scope"]="PREFILL_V2_LOAD_RUN_CLOSE_WALL";
    phases["decode"]["native_api"]="Run";phases["draft"]["native_api"]="Process";
#elif defined(QWEN35_RESIDENT_V1)
    timing["prefill_child_process_wall_ms"]=0;timing["prefill_child_processes_timed"]=0;
    timing["prefill_model_load_ms"]=prefill_load_ms;timing["prefill_process_ms"]=prefill_process_ms;timing["prefill_model_unload_ms"]=prefill_unload_ms;timing["prefill_v1_total_ms"]=prefill_total_ms;
    phases["prefill"]=phase(prefills,prefill_total_ms);phases["prefill"]["native_api"]="Process";phases["prefill"]["timing_scope"]="PREFILL_V1_LOAD_PROCESS_UNLOAD_WALL";
    phases["prompt_ingest"]["calls"]=prefills;phases["prompt_ingest"]["total_ms"]=prefill_total_ms;phases["prompt_ingest"]["mean_ms"]=Ratio(prefill_total_ms,prefills);phases["prompt_ingest"]["timing_scope"]="PREFILL_V1_LOAD_PROCESS_UNLOAD_WALL";
    for(auto key:{"decode","draft"})phases[key]["native_api"]="Process";
#else
    phases["prefill"]["timing_scope"]="CHILD_WALL_INCLUDES_LOAD_RUN_UNLOAD";phases["prompt_ingest"]["timing_scope"]="CHILD_WALL_INCLUDES_LOAD_RUN_UNLOAD";
#endif
    for(auto key:{"decode","draft"})phases[key]["timing_scope"]="RESIDENT_RUN_INCLUDES_TENSOR_COPIES_READBACK_HASH_AND_BUFFER_CHECKS_NOT_NPU_ONLY";
    if(c.fast){
      loop["runtime_profile"]="compact-memory-v1";
      loop["diagnostics"]="DISABLED_TENSOR_HASH_READBACK_AND_FP16_SCANS";
      loop["intermediate_tensor_io"]="NONE_MEMORY_ONLY";
      loop["startup_payload_validation"]=c.size_only?"SIZE_AND_ABI_ONLY":"OWNED_CACHE_EXTENTS_NEW_UPLOAD_SHA256";
      for(auto key:{"decode","draft"})phases[key]["timing_scope"]="RESIDENT_MEMORY_RUN_WITH_TENSOR_COPIES_NO_HASH_OR_FILE_IO";
    }
#ifdef QWEN35_RESIDENT_MIXED
    if(c.draft_preread)loop["draft_preread"]="sequential-1m-v1";
    J api=J::Obj();api["timing_scope"]="SYNCHRONOUS_API_WALL_NOT_NPU_ONLY";
    auto api_phase=[](std::uint64_t calls,std::uint64_t us,const char* name){J p=J::Obj();p["calls"]=calls;p["total_us"]=us;p["mean_ms"]=calls?Ratio(us,calls*1000):J();p["native_api"]=name;return p;};
    api["prefill"]=api_phase(prefills,prefill_api_us,"Run");api["decode"]=api_phase(e.target_runs,e.target_api_us,"Run");api["draft"]=api_phase(e.draft_runs,e.draft_api_us,"Process");loop["native_api_latency"]=api;
#endif
    J life=J::Obj();for(auto key:{"launched_child_count","completed_child_count","active_child_count","last_child_pid","last_child_exitcode","child_status"})life[key]=progress.at(key);life["children"]=children;loop["child_lifecycle"]=life;
    progress["resident_lifecycle"]=loop["resident_lifecycle"];
  }
  void StartResident() {
#if defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
    Require(prefill_lifecycle.closed&&!prefill_executor&&prefill_lifecycle.cleanup_error.empty(),"prefill model must close successfully before resident models load");
#endif
    Require(prefills==1&&targets==0&&drafts==0,"resident managers must load after the sole prefill completes");Pins();stage="resident-load";Progress();Lifecycle("resident-load-start","session="+session_id);
    qwen35::dflash::ResidentSessionConfig config;config.target_model=c.files.at("target_model").path;config.target_weight_directory=config.target_model.parent_path();config.draft_model=c.files.at("draft_model").path;
    config.runtime_version=c.strings.at("runtime_version");config.target_inputs=c.resident_inputs;config.target_outputs=c.resident_outputs;config.timeout_ms=c.timeout;config.model_load_api=kResidentLoadApi;config.diagnostics=!c.fast;
    config.draft_preread=c.draft_preread;
    if(c.incremental){config.incremental_draft=c.incremental_abi;draft_cache=std::make_unique<qwen35::dflash::IncrementalDraftCache>(c.incremental_abi);}
    MemorySnapshot("before-load");session.Load(config);MemorySnapshot("after-both-load");Pins();Lifecycle("resident-load-end","session="+session_id);stage="resident-ready";Progress();
  }
  std::uint64_t BeginResidentCall(const std::string& kind) {
    const auto index=resident_call_index++;progress["resident_call_status"]="RUNNING";progress["resident_call_kind"]=kind;progress["resident_call_index"]=index;stage="resident-"+kind+"-run";Progress();if(!c.fast)Lifecycle("resident-run-start","kind="+kind+" index="+std::to_string(index));return index;
  }
  void EndResidentCall(const std::string& kind,std::uint64_t index,std::uint64_t wall_ms) {
    progress["resident_call_status"]="RETURNED";if(!c.fast)Lifecycle("resident-run-end","kind="+kind+" index="+std::to_string(index)+" wall_ms="+std::to_string(wall_ms));
  }
  J IoEvidence(const qwen35::dflash::HiaiIoDiagnostic& diag) {
    J result=J::Obj();result["enabled"]=true;result["process_called"]=diag.process_called;result["output_fill_byte"]=diag.output_fill_byte;result["input_readback_point"]="AFTER_ALL_INPUT_COPIES_AND_OUTPUT_FILL_BEFORE_PROCESS";result["inputs"]=J::Arr();result["outputs"]=J::Arr();
    for(const auto& row:diag.inputs){J value=J::Obj();value["index"]=row.index;value["logical_bytes"]=row.logical_bytes;value["allocated_bytes"]=row.allocated_bytes;value["readback_sha256"]=row.readback_sha256;value["readback_matches_source"]=row.readback_matches_source;result["inputs"].a.push_back(value);}
    for(const auto& row:diag.outputs){J value=J::Obj();value["index"]=row.index;value["logical_bytes"]=row.logical_bytes;value["checked_after_process"]=row.checked_after_process;value["object_same"]=row.object_same;value["buffer_same"]=row.buffer_same;result["outputs"].a.push_back(value);}return result;
  }
  void ValidateResidentIo(const qwen35::dflash::HiaiIoDiagnostic& diag,const std::vector<Bytes>& data,std::size_t count) {
    Require(diag.process_called&&diag.output_fill_byte==0&&diag.inputs.size()==data.size()&&diag.outputs.size()==count,"resident actual input/output diagnostics incomplete");
    for(std::size_t i=0;i<data.size();++i)Require(diag.inputs[i].index==i&&diag.inputs[i].logical_bytes==data[i].size()&&diag.inputs[i].readback_matches_source&&diag.inputs[i].readback_sha256==Hash(data[i]),"resident actual copied input bytes differ");
    for(std::size_t i=0;i<count;++i)Require(diag.outputs[i].index==i&&diag.outputs[i].checked_after_process&&diag.outputs[i].object_same&&diag.outputs[i].buffer_same,"resident output buffer identity differs");
  }
#endif
  void Pins(){if(c.fast)return;for(const auto& x:c.files)Stable(x.second);Stable(c.config);for(const auto& x:kept)Stable(x.second);}
  void Keep(const File& f){
    if(!c.dump&&f.path!=out/"loop.json"&&f.path!=out/"progress.json")return;
    Stable(f);auto rel=fs::relative(f.path,out).generic_string();
    if(kept.count(rel)){const auto& old=kept.at(rel);Require(old.stamp==f.stamp&&old.hash==f.hash&&old.bytes==f.bytes,"retained observation differs: "+f.path.string());Stable(old);return;}
    kept.emplace(rel,f);retrieve.insert(rel);
  }
  void Keep(const fs::path& p){if(!c.dump&&p!=out/"loop.json"&&p!=out/"progress.json")return;auto rel=fs::relative(p,out).generic_string();if(kept.count(rel)){Stable(kept.at(rel));return;}Keep(Observe(p,Stat(p).size));}
  void KeepLogs(const fs::path& p){if(!c.dump)return;for(auto name:{"tool.stdout","tool.stderr","tool.pid","tool.exitcode","command.json"})if(fs::exists(p/name))Keep(p/name);}
  void KvCounts(){
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
    if(c.incremental) {
      J kv=J::Obj();kv["variant"]="incremental-context-kv-v1";kv["context_rows"]=c.incremental_abi.context_rows;
      kv["cache_capacity"]=c.incremental_abi.capacity;kv["max_draft_tokens"]=c.incremental_abi.candidates;
      kv["cached_context_rows"]=draft_cache?draft_cache->cursor():0;
      kv["cache_updates"]=draft_cache?draft_cache->calls():0;
      kv["catchup_calls"]=draft_cache?draft_cache->catchup_calls():0;
      kv["proposal_calls"]=draft_cache?draft_cache->rounds():0;
      kv["cache_policy"]="COMMITTED_CONTEXT_ONLY_ANCHOR_AND_PROPOSALS_EXCLUDED";
      kv["state_storage"]="SDK_INPUT_TENSORS_DELTA_WRITEBACK";
      const auto& evidence=session.evidence();
      kv["cache_commit_bytes"]=evidence.draft_kv_commit_bytes;kv["cache_commit_us"]=evidence.draft_kv_commit_us;
      kv["cache_commits"]=evidence.draft_kv_commits;kv["committed_rows"]=evidence.draft_kv_rows;
      kv["cache_host_staging_bytes"]=0;kv["hidden_readback_bytes"]=0;
      kv["numerical_validation"]="NOT_EVALUATED";loop["draft_kv"]=kv;
    }
#endif
  }
  void Counts(){KvCounts();J n=J::Obj();n["target_native_calls"]=targets;n["draft_native_calls"]=drafts;if(c.prefill)n["prefill_native_calls"]=prefills;n["considered"]=considered;n["accepted"]=accepted;n["corrections"]=corrections;n["bonuses"]=bonuses;loop["counts"]=n;loop["prompt"]=Array(c.prompt);loop["generated_tokens"]=Array(generated);loop["full_prefix"]=Array(prefix);loop["prompt_token_ids"]=Array(c.prompt);loop["generated_token_ids"]=Array(generated);loop["full_prefix_token_ids"]=Array(prefix);J timing=J::Obj();timing["scope"]="MONOTONIC_WALL_CLOCK_NOT_NPU_INFERENCE";timing["startup_source_hash_ms"]=c.startup_hash_ms;timing["target_child_process_wall_ms"]=target_child_ms;timing["draft_child_process_wall_ms"]=draft_child_ms;timing["target_child_processes_timed"]=targets;timing["draft_child_processes_timed"]=drafts;timing["total_wall_ms"]=ElapsedMs(c.started);if(c.prefill){timing["prefill_child_process_wall_ms"]=prefill_child_ms;timing["prefill_child_processes_timed"]=prefills;}loop["supervisor_timing"]=timing;J acceptance=J::Obj();acceptance["accepted"]=accepted;acceptance["considered"]=considered;acceptance["rate"]=Ratio(accepted,considered);loop["acceptance"]=acceptance;J phases=J::Obj();phases["timing_scope"]="CHILD_WALL_INCLUDES_LOAD_RUN_UNLOAD";phases["inference_only_status"]="NOT_PROVIDED";auto phase=[](std::uint64_t calls,std::uint64_t ms){J row=J::Obj();row["calls"]=calls;row["total_ms"]=ms;row["mean_ms"]=Ratio(ms,calls);return row;};phases["prompt_ingest"]=phase(c.prefill?prefills:prompt_calls,c.prefill?prefill_child_ms:prompt_child_ms);phases["prompt_ingest"]["mechanism"]=c.prefill?"PREFILL64_PADDED_SHORT_PROMPT_EXPERIMENTAL":"DECODE1_SERIAL_PROMPT_INGESTION";if(c.prefill){phases["prefill"]=phase(prefills,prefill_child_ms);phases["prompt_ingest"]["latency_alias"]="prefill";}phases["decode"]=phase(targets-prompt_calls,target_child_ms-prompt_child_ms);phases["draft"]=phase(drafts,draft_child_ms);loop["phase_latency"]=phases;if(!c.dump){loop.o.erase("prompt_token_ids");loop.o.erase("generated_token_ids");loop.o.erase("full_prefix_token_ids");loop["completed_rounds"]=round_count;J life=J::Obj();for(auto key:{"launched_child_count","completed_child_count","active_child_count","last_child_pid","last_child_exitcode","child_status"})life[key]=progress.at(key);life["children"]=children;loop["child_lifecycle"]=life;}}
  void Progress(){
    if(c.fast&&stage!="initialize"&&stage!="resident-load"&&stage!="resident-ready"&&stage!="resident-close"&&stage!="prefill-v2-load"&&stage!="prefill-v2-close"&&stage!="complete"&&progress.at("status").str()!="FAILED")return;
    Counts();
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
    ResidentCounts();
#endif
    progress["stage"]=stage;progress["committed_prefix_length"]=prefix.size();progress["generated_tokens"]=Array(generated);progress["counts"]=loop["counts"];progress["supervisor_timing"]=loop["supervisor_timing"];Atomic(out/"progress.json",progress);}
  void PruneOwned(const fs::path& tree){
#if defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
    Require(!c.dump&&progress.at("child_status").str()=="NONE"&&progress.at("active_child_count").num()==0&&prefill_lifecycle.closed&&prefill_lifecycle.cleanup_error.empty(),"direct scratch cleanup requires closed successful prefill and no child process");
    Require(progress.at("resident_call_status").str()=="RETURNED","direct scratch cleanup requires a returned synchronous model call");
#else
    Require(!c.dump&&progress.at("child_status").str()=="EXITED"&&progress.at("last_child_exitcode").num()==0&&progress.at("active_child_count").num()==0,"scratch cleanup requires a known successful child exit");
#endif
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
    if(tree.lexically_relative(out).generic_string().rfind("prefill/",0)!=0||targets)
      Require(progress.at("resident_call_status").str()=="RETURNED","scratch cleanup requires a returned resident call");
#endif
    const auto relative=tree.lexically_relative(out).generic_string();
    Require(std::regex_match(relative,std::regex("target/step_[0-9]{4}(/inputs)?|prefill/step_0000(/inputs)?|draft/call-[0-9]{4}")),"cleanup outside exact owned scratch");
    Safe(tree,true);std::vector<fs::path> dirs{tree};std::vector<std::pair<fs::path,Stamp>> files;
    for(const auto& e:fs::recursive_directory_iterator(tree)){
      struct stat s{};Require(::lstat(e.path().c_str(),&s)==0&&!S_ISLNK(s.st_mode),"unsafe owned scratch entry");
      if(S_ISDIR(s.st_mode)){Safe(e.path(),true);dirs.push_back(e.path());}
      else{Require(S_ISREG(s.st_mode)&&s.st_nlink==1,"scratch cleanup rejects nonregular/shared entry");files.emplace_back(e.path(),Stat(e.path()));}
      Require(files.size()+dirs.size()<=4096,"owned scratch cleanup entry budget exceeded");
    }
    // Validate the complete tree before deleting any entry. Do not follow links
    // or sweep cache directories; only this successfully consumed attempt.
    for(const auto& f:files)Require(Stat(f.first)==f.second,"scratch changed before cleanup");
    for(const auto& f:files){Require(Stat(f.first)==f.second,"scratch changed during cleanup");Require(::unlink(f.first.c_str())==0,"owned scratch unlink failed");}
    std::sort(dirs.begin(),dirs.end(),[](const fs::path& a,const fs::path& b){return a.native().size()>b.native().size();});
    for(const auto& d:dirs){Safe(d,true);Require(::rmdir(d.c_str())==0,"owned scratch directory not empty/changed");}
  }
  void PruneDraft(const fs::path& call,const fs::path& linked){
    Require(progress.at("child_status").str()=="EXITED"&&progress.at("last_child_exitcode").num()==0&&progress.at("active_child_count").num()==0,"draft scratch cleanup requires known success");
    auto& model=c.files.at("draft_model");Stable(model);auto before=model.stamp;
    Require(Stat(linked)==before,"owned draft model hardlink identity changed before cleanup");
    Require(::unlink(linked.c_str())==0,"owned draft model hardlink unlink failed");auto after=Stat(model.path);
    Require(before.dev==after.dev&&before.ino==after.ino&&before.size==after.size&&before.mt==after.mt&&before.mode==after.mode&&before.uid==after.uid&&before.gid==after.gid&&before.links==after.links+1,"draft unlink changed source beyond owned link metadata");model.stamp=after;
    PruneOwned(call);Pins();
  }
  void Manifest(){std::string t="QWEN35_BOARD_SERIAL_RETRIEVAL\t1\n";for(const auto& s:retrieve){auto p=out/s;const auto& old=kept.at(s);Stable(old);auto f=Observe(p,old.bytes,old.hash);Require(f.stamp==old.stamp,"retrieval source changed");t+="file\t"+s+"\t"+f.hash+"\t"+std::to_string(f.bytes)+"\n";}auto tmp=out/"retrieval-manifest.tsv.new";Write(tmp,t);Require(::rename(tmp.c_str(),(out/"retrieval-manifest.tsv").c_str())==0,"manifest rename failed");}
  void Prefill(){
    Require(c.prefill&&prefills==0&&targets==0&&prefix.empty()&&features.empty()&&states.empty(),"prefill must be the single first model call");stage="prefill-inputs";Pins();
    auto step=out/"prefill/step_0000";if(!c.fast){NewDir(step);NewDir(step/"inputs");NewDir(step/"outputs");}inputs.clear();
    const auto valid=c.prompt.size();Bytes embeddings(64*2048,0),scales(64*4,0),positions(64*4,0),conv(512,0),mask(64*8192,0);
    for(std::size_t i=0;i<valid;++i){auto row=EmbeddingRow(false,c.prompt[i]),scale=EmbeddingRow(true,c.prompt[i]);std::uint32_t bits=0;for(int n=0;n<4;++n)bits|=std::uint32_t(scale[n])<<(8*n);float value;std::memcpy(&value,&bits,4);Require(std::isfinite(value)&&value>=0,"invalid original prefill embedding scale");std::copy(row.begin(),row.end(),embeddings.begin()+i*2048);std::copy(scale.begin(),scale.end(),scales.begin()+i*4);}
    for(std::size_t q=0;q<64;++q){auto word=Scalar(q,4);std::copy(word.begin(),word.end(),positions.begin()+q*4);for(std::size_t key=0;key<2048;++key){const bool allowed=q<valid?key<=q:key==0;if(!allowed){mask[q*8192+key*4+2]=0x80;mask[q*8192+key*4+3]=0xff;}}}
    auto length=Scalar(valid,4);std::copy(length.begin(),length.end(),conv.begin());
    std::map<std::string,Bytes> controls={{"arg1_1",std::move(embeddings)},{"arg0_1",std::move(scales)},{"arg107_1",positions},{"arg2_1",positions},{"arg1238_1",Scalar(valid-1,4)},{"arg114_1",Scalar(valid,8)},{"arg18_1",Scalar(valid,2)},{"arg14_1",std::move(conv)},{"arg113_1",std::move(mask)}};
    std::set<unsigned> state_inputs;for(const auto& s:c.prefill_states)state_inputs.insert(s.in);
    J inputrows=J::Arr(),outputrows=J::Arr();std::string paths,types,outtypes;
#if defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
    std::vector<Bytes> prefill_data;
#endif
    for(const auto& t:c.prefill_inputs){auto path=step/"inputs"/(Numbered("input_",t.index)+".bin");Bytes data=state_inputs.count(t.index)?Bytes(t.bytes,0):controls.at(t.name);Require(data.size()==t.bytes,"prefill input extent mismatch");
#ifdef QWEN35_RESIDENT_MIXED
      if(c.fast){prefill_data.push_back(std::move(data));continue;}
#endif
      Write(path,data);auto f=Observe(path,t.bytes,Hash(data));inputs.push_back(f);J row=Record(f,out);row["index"]=t.index;row["name"]=t.name;row["tool_type"]=t.type;inputrows.a.push_back(row);
#if defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
      prefill_data.push_back(std::move(data));
#endif
      if(!paths.empty()){paths+=',';types+=',';}paths+=path.string();types+=t.type;}
    for(const auto& t:c.prefill_outputs){if(!outtypes.empty())outtypes+=',';outtypes+=t.type;}
    for(const auto& f:inputs){Stable(f);}Pins();
#if defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
#ifdef QWEN35_RESIDENT_MIXED
    const std::string prefill_api_stage="prefill-v2";
#else
    const std::string prefill_api_stage="prefill-v1";
#endif
    stage=prefill_api_stage+"-load";Progress();Lifecycle(prefill_api_stage+"-load-start","model=qwen35_resident_prefill");
    std::vector<Bytes> prefill_memory(82);qwen35::dflash::HiaiIoDiagnostic prefill_diag;std::map<unsigned,fs::path> mapping;
    std::uint64_t prefill_call_index=0;auto prefill_begin=Clock::now();
    try {
#ifdef QWEN35_RESIDENT_MIXED
      MemorySnapshot("before-prefill-load");
#endif
      auto load_begin=Clock::now();
      try{
#ifdef QWEN35_RESIDENT_MIXED
        prefill_executor=std::make_unique<qwen35::dflash::HiaiOriginalTargetExecutor>(c.files.at("prefill_model").path,c.files.at("prefill_model").path.parent_path(),c.strings.at("runtime_version"),c.prefill_bound_inputs,c.prefill_bound_outputs,"qwen35_resident_prefill",&prefill_lifecycle,qwen35::dflash::HiaiOriginalTargetCleanupPolicy::ManagerDeinitOnly,!c.fast);
#else
        prefill_executor=std::make_unique<qwen35::dflash::HiaiOriginalTargetV1Executor>(c.files.at("prefill_model").path,c.files.at("prefill_model").path.parent_path(),c.strings.at("runtime_version"),c.prefill_bound_inputs,c.prefill_bound_outputs,c.timeout,"qwen35_resident_prefill",&prefill_lifecycle);
#endif
      }
      catch(...){prefill_load_ms+=ElapsedMs(load_begin);throw;}
      prefill_load_ms+=ElapsedMs(load_begin);Lifecycle(prefill_api_stage+"-load-end","model=qwen35_resident_prefill");
#ifdef QWEN35_RESIDENT_MIXED
      MemorySnapshot("after-prefill-load");
#endif
      prefill_call_index=BeginResidentCall("prefill");auto process_begin=Clock::now();
      try{for(std::size_t i=0;i<prefill_data.size();++i)prefill_executor->SetInput(i,prefill_data[i]);prefill_executor->Execute();++prefills;
        #ifdef QWEN35_RESIDENT_MIXED
        prefill_api_us+=prefill_executor->last_api_us();
#endif
        if(!c.fast){prefill_diag=prefill_executor->io_diagnostic();ValidateResidentIo(prefill_diag,prefill_data,82);}
        for(const auto& t:c.prefill_outputs){auto bytes=prefill_executor->CopyOutput(t.index);Require(bytes.size()==t.bytes,"direct prefill output extent differs");if(c.fast){prefill_memory[t.index]=std::move(bytes);continue;}auto path=step/"outputs"/Numbered("output_",t.index);Write(path,bytes);mapping.emplace(t.index,path);}
      }catch(...){prefill_process_ms+=ElapsedMs(process_begin);throw;}
      prefill_process_ms+=ElapsedMs(process_begin);EndResidentCall("prefill",prefill_call_index,prefill_process_ms);
      stage=prefill_api_stage+"-close";Progress();ClosePrefill();Require(prefill_lifecycle.closed&&prefill_lifecycle.cleanup_error.empty(),"direct prefill failed to close successfully");
#ifdef QWEN35_RESIDENT_MIXED
      MemorySnapshot("after-prefill-close");
#endif
    }catch(...){try{ClosePrefill();}catch(const std::exception& cleanup){loop["prefill_cleanup_error"]=cleanup.what();}prefill_total_ms=ElapsedMs(prefill_begin);throw;}
    prefill_total_ms=ElapsedMs(prefill_begin);Lifecycle(prefill_api_stage+"-close-end","model=qwen35_resident_prefill");
    if(c.fast){
      next=Argmax(prefill_memory[0]);features=std::move(prefill_memory[81]);features.resize(valid*kFeatureBytes);
      for(const auto& decode:c.states){auto found=std::find_if(c.prefill_states.begin(),c.prefill_states.end(),[&](const auto& state){return state.decode_ref==decode.ref;});
        Require(found!=c.prefill_states.end()&&prefill_memory[found->out].size()==decode.bytes,"prefill memory handoff extent differs");memory_states[decode.in]=std::move(prefill_memory[found->out]);}
      prefix=c.prompt;stage="prefill-committed";Progress();return;
    }
    for(const auto& f:inputs){Stable(f);Require(Observe(f.path,f.bytes).hash==f.hash,"direct prefill changed private input bytes");}Pins();stage="prefill-bind-outputs";
#else
    for(const auto& f:inputs){Stable(f);}Pins();stage="prefill-child";Progress();
    std::vector<std::string> args={c.files.at("target_tool").path.string(),"--model="+c.files.at("prefill_model").path.string(),"--input="+paths,"--times=1","--modelmanager_itf=V2","--weight_dir="+c.files.at("prefill_model").path.parent_path().string(),"--enable_item=1","--direct_from_file=true","--input_type="+types,"--output_type="+outtypes,"--output_dir="+(step/"outputs").string()};
    auto child=Run(args,step,c.strings.at("target_library_path"),step,progress,children,out);++prefills;prefill_child_ms+=child.wall_ms;KeepLogs(step);Require(child.exitcode==0,"prefill child failed; no retry");
    for(const auto& f:inputs){Stable(f);Require(Observe(f.path,f.bytes).hash==f.hash,"prefill changed private input bytes");}Pins();stage="prefill-bind-outputs";
    auto log=Text(step/"tool.stdout",1<<20);std::map<unsigned,fs::path> mapping;std::istringstream ls(log);std::string line;std::regex pattern("^output\\[([0-9]+)\\][ \\t]*:[ \\t]*(/[^ \\t\\r]+)[ \\t\\r]*$");
    while(std::getline(ls,line)){std::smatch m;if(std::regex_match(line,m,pattern)){auto index=Integer(m[1],81);Require(!mapping.count(index),"duplicate logged prefill output");auto p=Absolute(m[2]);Require(p.parent_path()==step/"outputs","prefill output outside owned attempt");mapping.emplace(index,p);}else if(line.find("output[")!=std::string::npos)Fail("malformed prefill output log mapping");}
    Require(mapping.size()==82&&log.find("Execute result :Success")!=std::string::npos,"prefill log missing complete explicit output mapping/success");
#endif
    std::set<fs::path> actual,selected;for(const auto& e:fs::directory_iterator(step/"outputs")){Safe(e.path());actual.insert(e.path());}Require(actual.size()==82,"prefill output directory count differs");outputs.clear();Bytes logits,feature_rows;
    for(const auto& t:c.prefill_outputs){auto path=mapping.at(t.index);Require(actual.count(path)&&selected.insert(path).second,"duplicate/absent prefill output file");File f;if(t.index==0||t.index==81){auto observed=ReadObserved(path,t.bytes);f=observed.file;(t.index==0?logits:feature_rows)=std::move(observed.bytes);}else f=Observe(path,t.bytes);outputs.push_back(f);J row=Record(f,out);row["index"]=t.index;row["runtime_name"]=t.name;row["tool_type"]=t.type;outputrows.a.push_back(row);Keep(f);}
    next=Argmax(logits);Require(feature_rows.size()==64*kFeatureBytes,"prefill feature physical64 extent mismatch");feature_rows.resize(valid*kFeatureBytes);FiniteHalf(feature_rows);
    std::map<std::string,PrefillState> bridge;for(const auto& s:c.prefill_states)bridge.emplace(s.decode_ref,s);J state_rows=J::Arr();std::vector<File> newstates;
    for(const auto& decode:c.states){const auto& s=bridge.at(decode.ref);const auto& f=outputs.at(s.out);Stable(f);Require(f.bytes==decode.bytes,"prefill state handoff byte extent differs");newstates.push_back(f);J row=J::Obj();row["decode_source_refdata"]=decode.ref;row["decode_runtime_input_index"]=decode.in;row["decode_runtime_output_index"]=decode.out;row["prefill_source_refdata"]=s.ref;row["prefill_runtime_input_index"]=s.in;row["prefill_runtime_output_index"]=s.out;row["bytes"]=s.bytes;row["prefill_output"]=Record(f,out);row["initialization"]="ZERO_FP16";state_rows.a.push_back(row);}
    if(c.dump){J receipt=J::Obj();receipt["schema_version"]=1;receipt["artifact_kind"]="qwen35-board-serial-prefill-step";receipt["evidence_kind"]=kEvidence;receipt["status"]="EXECUTED_OUTPUTS_BOUND";
#if defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
      receipt["schema_version"]=2;ResidentIdentity(receipt,prefill_call_index);receipt["io_diagnostic"]=IoEvidence(prefill_diag);
#ifdef QWEN35_RESIDENT_MIXED
      receipt["prefill_lifecycle"]=V2Lifecycle(prefill_lifecycle);receipt["model_load_api"]="file-v2";
#else
      receipt["prefill_lifecycle"]=LegacyLifecycle(prefill_lifecycle);LegacyAbiEvidence(receipt,prefill_lifecycle);
#endif
#else
      receipt["pid"]=child.pid;receipt["exitcode"]=0;
#endif
      receipt["model_sha256"]=c.files.at("prefill_model").hash;receipt["prompt_token_ids"]=Array(c.prompt);receipt["logical_prompt_length"]=valid;receipt["physical_sequence_length"]=64;receipt["selected_output_pos"]=valid-1;receipt["control_semantics"]="EXPERIMENTAL_PADDED_PREFILL_VALID_LENGTH_NOT_NUMERICALLY_VALIDATED";receipt["inputs"]=inputrows;receipt["outputs"]=outputrows;receipt["state_mapping"]=state_rows;receipt["next_token_id"]=next;J feature=J::Obj();feature["source"]=Record(outputs.at(81),out);feature["row_start"]=0;feature["row_count"]=valid;feature["bytes"]=feature_rows.size();feature["sha256"]=Hash(feature_rows);receipt["feature_prefix"]=feature;Write(step/"receipt.json",Dump(receipt)+"\n");auto file=Observe(step/"receipt.json",Stat(step/"receipt.json").size);Keep(file);J event=J::Obj();event["receipt"]=Record(file,out);
#if defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
      ResidentIdentity(event,prefill_call_index);
#else
      event["pid"]=child.pid;
#endif
      event["prompt_length"]=valid;event["next_token_id"]=next;loop["prefill_steps"].a.push_back(event);}
    for(const auto& f:outputs){Stable(f);}for(const auto& f:inputs){Stable(f);}Pins();states=std::move(newstates);features=std::move(feature_rows);prefix=c.prompt;
    if(!c.dump){PruneOwned(step/"inputs");inputs.clear();last_target_step=step;}stage="prefill-committed";Progress();
  }
  void Target(std::uint32_t token,const std::string& role){stage="target-inputs";Pins();for(const auto& f:states)Stable(f);auto step=out/"target"/Numbered("step_",targets);if(!c.fast){NewDir(step);NewDir(step/"inputs");NewDir(step/"outputs");}auto oldstates=states;inputs.clear();
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
    std::vector<Bytes> resident_inputs;
#endif
    Bytes row=EmbeddingRow(false,token),scale=EmbeddingRow(true,token);std::uint32_t sb=0;for(int i=0;i<4;++i)sb|=std::uint32_t(scale[i])<<(8*i);float sf;std::memcpy(&sf,&sb,4);Require(std::isfinite(sf)&&sf>=0,"invalid original embedding scale");
    std::map<std::string,Bytes> control={{"arg1_1",row},{"arg0_1",scale},{"arg107_1",Scalar(prefix.size(),4)},{"arg2_1",Scalar(prefix.size(),4)},{"arg1238_1",Scalar(0,4)},{"arg114_1",Scalar(1,8)},{"arg18_1",Scalar(1,2)},{"arg14_1",Bytes(512,0)},{"arg113_1",Bytes(8192,0)}};
    control.at("arg14_1")[0]=1;for(std::size_t i=prefix.size()+1;i<2048;++i){control.at("arg113_1")[i*4+2]=0x80;control.at("arg113_1")[i*4+3]=0xff;}
#ifdef QWEN35_RESIDENT_MIXED
    if(c.fast){
      std::vector<Bytes> data;data.reserve(89);
      for(const auto& t:c.inputs){auto found=control.find(t.name);data.push_back(found==control.end()?std::move(memory_states[t.index]):std::move(found->second));}
      const auto index=BeginResidentCall("target");auto actual=session.RunTarget(data);EndResidentCall("target",index,actual.wall_ms);++targets;
      next=Argmax(actual.outputs.at(0));const auto& chunk=actual.outputs.at(81);features.insert(features.end(),chunk.begin(),chunk.end());
      for(const auto& state:c.states)memory_states[state.in]=std::move(actual.outputs.at(state.out));
      prefix.push_back(token);stage="target-committed";Progress();return;
    }
#endif
    std::map<unsigned,std::size_t> byinput;for(std::size_t i=0;i<c.states.size();++i)byinput[c.states[i].in]=i;
    J inputrows=J::Arr(),feedback=J::Arr();std::string paths,types,outtypes;
    for(const auto& t:c.inputs){auto path=step/"inputs"/(Numbered("input_",t.index)+".bin");Bytes b;auto found=byinput.find(t.index);if(found==byinput.end())b=control.at(t.name);else if(oldstates.empty())b=Bytes(t.bytes,0);else b=Read(oldstates[found->second].path,t.bytes);
      Require(b.size()==t.bytes,"target input extent mismatch");
      std::string expected_hash;
      if(found==byinput.end())expected_hash=Hash(b);
      else if(!oldstates.empty()){Stable(oldstates[found->second]);expected_hash=oldstates[found->second].hash;}
      else {auto known=zero_hashes.find(t.bytes);if(known==zero_hashes.end())known=zero_hashes.emplace(t.bytes,Hash(b)).first;expected_hash=known->second;}
      Write(path,b);auto f=Observe(path,t.bytes,expected_hash);inputs.push_back(f);J r=Record(f,out);r["index"]=t.index;r["name"]=t.name;r["tool_type"]=t.type;inputrows.a.push_back(r);
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
      resident_inputs.push_back(std::move(b));
#endif
      if(found!=byinput.end()){J state=J::Obj();state["source_refdata"]=c.states[found->second].ref;state["runtime_input_index"]=t.index;state["runtime_output_index"]=c.states[found->second].out;state["input"]=r;if(!oldstates.empty()){Stable(oldstates[found->second]);Require(f.hash==oldstates[found->second].hash,"state input differs from committed parent");state["parent_output"]=Record(oldstates[found->second],out);}else state["initialization"]="ZERO_FP16";feedback.a.push_back(state);}
      if(!paths.empty()){paths+=',';types+=',';}paths+=path.string();types+=t.type;}
    for(const auto& t:c.outputs){if(!outtypes.empty())outtypes+=',';outtypes+=t.type;}
    for(const auto& f:inputs)Stable(f);
    Pins();stage="target-child";Progress();
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
    const auto call_index=BeginResidentCall("target");auto actual_run=session.RunTarget(resident_inputs);EndResidentCall("target",call_index,actual_run.wall_ms);++targets;
    ValidateResidentIo(actual_run.io_diagnostic,resident_inputs,82);Require(actual_run.outputs.size()==82,"resident target output count differs");
    std::map<unsigned,fs::path> mapping;
    for(const auto& t:c.outputs){Require(actual_run.outputs[t.index].size()==t.bytes,"resident target output extent differs");auto path=step/"outputs"/Numbered("output_",t.index);Write(path,actual_run.outputs[t.index]);mapping.emplace(t.index,path);}
    for(const auto& f:inputs){Stable(f);Require(Observe(f.path,f.bytes).hash==f.hash,"resident target changed private input bytes");}for(const auto& f:oldstates)Stable(f);Pins();stage="target-bind-outputs";
#else
    std::vector<std::string> args={c.files.at("target_tool").path.string(),"--model="+c.files.at("target_model").path.string(),"--input="+paths,"--times=1","--modelmanager_itf=V2","--weight_dir="+c.files.at("target_model").path.parent_path().string(),"--enable_item=1","--direct_from_file=true","--input_type="+types,"--output_type="+outtypes,"--output_dir="+(step/"outputs").string()};
    auto child=Run(args,step,c.strings.at("target_library_path"),step,progress,children,out);++targets;target_child_ms+=child.wall_ms;if(role=="prompt"){++prompt_calls;prompt_child_ms+=child.wall_ms;}KeepLogs(step);Require(child.exitcode==0,"target child failed; no retry");
    for(const auto& f:inputs){Stable(f);Require(Observe(f.path,f.bytes).hash==f.hash,"target changed private input bytes");}for(const auto& f:oldstates)Stable(f);Pins();
    stage="target-bind-outputs";auto log=Text(step/"tool.stdout",1<<20);std::map<unsigned,fs::path> mapping;std::istringstream ls(log);std::string line;
    std::regex pattern("^output\\[([0-9]+)\\][ \\t]*:[ \\t]*(/[^ \\t\\r]+)[ \\t\\r]*$");
    while(std::getline(ls,line)){std::smatch m;if(std::regex_match(line,m,pattern)){auto index=Integer(m[1],81);Require(!mapping.count(index),"duplicate logged target output");auto p=Absolute(m[2]);Require(p.parent_path()==step/"outputs","logged output outside owned attempt");mapping.emplace(index,p);}else if(line.find("output[")!=std::string::npos)Fail("malformed target output log mapping");}
    Require(mapping.size()==82&&log.find("Execute result :Success")!=std::string::npos,"target log missing complete explicit output mapping/success");
#endif
    std::set<fs::path> actual;for(const auto& e:fs::directory_iterator(step/"outputs")){Safe(e.path());actual.insert(e.path());}Require(actual.size()==82,"target output directory count differs");outputs.clear();J outputrows=J::Arr();std::set<fs::path> selected;
    Bytes logits,chunk;
    for(const auto& t:c.outputs){auto p=mapping.at(t.index);Require(actual.count(p)&&selected.insert(p).second,"duplicate or absent target file mapping");File f;
      if(t.index==0||t.index==81){auto observed=ReadObserved(p,t.bytes);f=observed.file;(t.index==0?logits:chunk)=std::move(observed.bytes);}else f=Observe(p,t.bytes);
      outputs.push_back(f);J r=Record(f,out);r["index"]=t.index;r["runtime_name"]=t.name;r["tool_type"]=t.type;outputrows.a.push_back(r);}
    next=Argmax(logits);FiniteHalf(chunk);
    if(c.dump){J receipt=J::Obj();receipt["schema_version"]=1;receipt["artifact_kind"]="qwen35-board-serial-target-step";receipt["evidence_kind"]=kEvidence;receipt["status"]="EXECUTED_OUTPUTS_BOUND";
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
    receipt["schema_version"]=2;ResidentIdentity(receipt,call_index);receipt["io_diagnostic"]=IoEvidence(actual_run.io_diagnostic);
#ifdef QWEN35_RESIDENT_V1
    LegacyAbiEvidence(receipt,session.evidence().target_v1_lifecycle);
#elif defined(QWEN35_RESIDENT_MIXED)
    receipt["model_load_api"]="file-v2";
#endif
#else
    receipt["pid"]=child.pid;receipt["exitcode"]=0;
#endif
    receipt["token_id"]=token;receipt["absolute_position"]=prefix.size();receipt["role"]=role;receipt["model_sha256"]=c.files.at("target_model").hash;receipt["inputs"]=inputrows;receipt["outputs"]=outputrows;receipt["state_feedback"]=feedback;receipt["next_token_id"]=next;receipt["control_semantics"]="EXPERIMENTAL_SINGLE_QUERY_ABSOLUTE_POSITION";
    Write(step/"receipt.json",Dump(receipt)+"\n");auto receiptfile=Observe(step/"receipt.json",Stat(step/"receipt.json").size);Keep(receiptfile);J event=J::Obj();event["receipt"]=Record(receiptfile,out);
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
    ResidentIdentity(event,call_index);
#else
    event["pid"]=child.pid;
#endif
    event["token_id"]=token;event["next_token_id"]=next;loop["target_steps"].a.push_back(event);}
    for(const auto& f:outputs){Stable(f);}for(const auto& f:inputs){Stable(f);}Pins();
    states.clear();for(const auto& s:c.states)states.push_back(outputs[s.out]);features.insert(features.end(),chunk.begin(),chunk.end());prefix.push_back(token);
    if(!c.dump){PruneOwned(step/"inputs");inputs.clear();if(!last_target_step.empty())PruneOwned(last_target_step);last_target_step=step;}stage="target-committed";Progress();
  }
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
  std::vector<std::uint32_t> IncrementalDraft() {
    Require(c.fast&&draft_cache,"incremental cache unavailable");stage="incremental-draft-inputs";Pins();
    std::vector<std::uint32_t> proposals;
    do {
      auto inputs=draft_cache->Prepare(features,prefix.size(),prefix.back());
      const auto index=BeginResidentCall("draft");auto actual=session.RunIncrementalDraft(inputs);
      EndResidentCall("draft",index,actual.wall_ms);++drafts;
      proposals=draft_cache->Complete(actual.outputs.at(0),[&]{session.CommitIncrementalDraft();});
    } while(proposals.empty());
    return proposals;
  }
  std::vector<std::uint32_t> ResidentDraft() {
    if(c.incremental)return IncrementalDraft();
    stage="draft-inputs";Pins();Require(!prefix.empty()&&prefix.size()<=64&&features.size()==prefix.size()*kFeatureBytes,"resident draft prefix/features mismatch");
    auto call=out/"draft"/Numbered("call-",drafts);auto bundle=call/"bundle";if(!c.fast){NewDir(call);NewDir(bundle);NewDir(call/"results");}
    std::vector<Bytes> data={Bytes(64*kFeatureBytes,0),Bytes(256,0),Bytes(256,0)};std::copy(features.begin(),features.end(),data[0].begin());
    for(std::size_t i=0;i<prefix.size();++i){auto value=Scalar(prefix[i],4);std::copy(value.begin(),value.end(),data[1].begin()+4*i);data[2][4*i]=1;}
    if(c.fast){
      const auto index=BeginResidentCall("draft");auto actual=session.RunDraft(data);EndResidentCall("draft",index,actual.wall_ms);++drafts;
      Require(actual.outputs.size()==1&&actual.outputs[0].size()==60,"draft top1 extent differs");std::vector<std::uint32_t> proposals;
      for(std::size_t i=0;i<15;++i){std::uint32_t token=0;for(int n=0;n<4;++n)token|=std::uint32_t(actual.outputs[0][i*4+n])<<(8*n);Require(token<kVocab,"draft token outside vocabulary");proposals.push_back(token);}return proposals;
    }
    const std::vector<std::string> names={"target_features","input_ids","attention_mask"},types={"float16","int32","int32"};
    std::vector<File> pins;J inputrows=J::Arr(),native_inputs=J::Arr();
    for(std::size_t i=0;i<data.size();++i){auto path=bundle/(Numbered("input_",i)+".bin");Write(path,data[i]);auto file=Observe(path,data[i].size(),Hash(data[i]));pins.push_back(file);J row=Record(file,out);row["index"]=i;row["name"]=names[i];inputrows.a.push_back(row);
      J native=row;native["path"]=path.filename().string();native["dtype"]=types[i];native["shape"]=i==0?Array({1,64,16384}):Array({1,64});native_inputs.a.push_back(native);}
    for(const auto& file:pins)Stable(file);Pins();auto index=BeginResidentCall("draft");auto actual=session.RunDraft(data);EndResidentCall("draft",index,actual.wall_ms);++drafts;
    ValidateResidentIo(actual.io_diagnostic,data,2);Require(actual.outputs.size()==2&&actual.outputs[0].size()==60&&actual.outputs[1].size()==61440,"resident draft output ABI differs");
    for(const auto& file:pins){Stable(file);Require(Observe(file.path,file.bytes).hash==file.hash,"resident draft input file bytes changed");}Pins();
    std::vector<File> produced;J native_outputs=J::Arr();for(std::size_t i=0;i<2;++i){auto path=call/"results"/(Numbered("output_",i)+".bin");Write(path,actual.outputs[i]);auto file=Observe(path,actual.outputs[i].size(),Hash(actual.outputs[i]));produced.push_back(file);Keep(file);J row=Record(file,out);row["index"]=i;row["name"]=i==0?"draft_top1":"draft_hidden";row["dtype"]=i==0?"int32":"float16";row["shape"]=i==0?Array({1,15}):Array({1,15,2048});row["path"]=path.filename().string();native_outputs.a.push_back(row);}
    FiniteHalf(actual.outputs[1]);std::vector<std::uint32_t> proposals;
    for(std::size_t i=0;i<15;++i){std::uint32_t token=0;for(int n=0;n<4;++n)token|=std::uint32_t(actual.outputs[0][i*4+n])<<(8*n);Require(token<kVocab,"resident draft token outside full vocabulary");proposals.push_back(token);}
    if(c.dump){J native=J::Obj();native["schema_version"]=2;native["artifact_kind"]="qwen35-resident-draft-call";native["status"]="EXECUTED";native["evidence_kind"]=kEvidence;ResidentIdentity(native,index);native["model_sha256"]=c.files.at("draft_model").hash;native["draft_abi_sha256"]=c.files.at("draft_abi").hash;
      native["runtime_version"]=c.strings.at("runtime_version");native["ddk_id"]=c.strings.at("ddk_id");native["firmware_id"]=c.strings.at("firmware_id");native["model_load_api"]=std::string(kResidentLoadApi)=="file-v2"?"file-v2":"legacy-buffer";native["sequence_length"]=64;native["target_calls"]=0;native["draft_calls"]=1;native["inputs"]=native_inputs;native["outputs"]=native_outputs;native["io_diagnostic"]=IoEvidence(actual.io_diagnostic);
#if defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
      LegacyAbiEvidence(native,session.evidence().draft_v1_lifecycle);
#endif
      Write(call/"results/result.json",Dump(native)+"\n");auto result=Observe(call/"results/result.json",Stat(call/"results/result.json").size);Keep(result);
      J receipt=J::Obj();receipt["schema_version"]=2;receipt["artifact_kind"]="qwen35-board-serial-draft-step";receipt["status"]="EXECUTED_OUTPUTS_BOUND";receipt["evidence_kind"]=kEvidence;ResidentIdentity(receipt,index);receipt["model_sha256"]=c.files.at("draft_model").hash;receipt["prefix"]=Array(prefix);receipt["feature_prefix_sha256"]=Hash(features);receipt["inputs"]=inputrows;receipt["native_receipt"]=Record(result,out);receipt["proposals"]=Array(proposals);
      Write(call/"receipt.json",Dump(receipt)+"\n");auto file=Observe(call/"receipt.json",Stat(call/"receipt.json").size);Keep(file);J event=J::Obj();event["receipt"]=Record(file,out);ResidentIdentity(event,index);event["proposals"]=Array(proposals);loop["draft_steps"].a.push_back(event);}
    for(const auto& file:produced)Stable(file);for(const auto& file:pins)Stable(file);Pins();if(!c.dump)PruneOwned(call);return proposals;
  }
#endif
  std::vector<std::uint32_t> Draft(){
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
    return ResidentDraft();
#else
    stage="draft-inputs";Pins();Require(!prefix.empty()&&prefix.size()<=64&&features.size()==prefix.size()*kFeatureBytes,"draft prefix/features mismatch");auto call=out/"draft"/Numbered("call-",drafts);NewDir(call);auto bundle=call/"bundle";NewDir(bundle);NewDir(bundle/"model");
    auto& model=c.files.at("draft_model");Stable(model);auto before=model.stamp;auto linked=bundle/"model/model.omc";Require(::link(model.path.c_str(),linked.c_str())==0,"board-local draft model hardlink failed");auto after=Stat(model.path);auto linkstat=Stat(linked);
    Require(before.dev==after.dev&&before.ino==after.ino&&before.size==after.size&&before.mt==after.mt&&before.mode==after.mode&&before.uid==after.uid&&before.gid==after.gid&&after.links==before.links+1&&linkstat==after,"hardlink changed original draft beyond owned link metadata");model.stamp=after;
    auto abi=Read(c.files.at("draft_abi").path,c.files.at("draft_abi").bytes);Write(bundle/"draft-abi.json",abi);Bytes padded(64*kFeatureBytes,0);std::copy(features.begin(),features.end(),padded.begin());Bytes ids(256,0),mask(256,0);for(std::size_t i=0;i<prefix.size();++i){auto b=Scalar(prefix[i],4);std::copy(b.begin(),b.end(),ids.begin()+4*i);mask[4*i]=1;}
    std::vector<std::string> names={"target_features","input_ids","attention_mask"},types={"float16","int32","int32"},shapes={"1,64,16384","1,64","1,64"};std::vector<Bytes> data={std::move(padded),std::move(ids),std::move(mask)};std::vector<File> pins;J inputrows=J::Arr();
    std::string wire="QWEN35_DRAFT_DIRECT_STEP\t1\nmodel\tmodel/model.omc\t"+model.hash+"\ndraft_abi\tdraft-abi.json\t"+c.files.at("draft_abi").hash+"\ntokens\t64\t0\n";
    for(std::size_t i=0;i<3;++i){auto name=Numbered("input_",i)+".bin";Write(bundle/name,data[i]);auto f=Observe(bundle/name,data[i].size(),Hash(data[i]));pins.push_back(f);wire+="input\t"+std::to_string(i)+"\t"+names[i]+"\t"+types[i]+"\t"+shapes[i]+"\t"+name+"\t"+f.hash+"\n";J r=Record(f,out);r["index"]=i;r["name"]=names[i];inputrows.a.push_back(r);}
    wire+="output\t0\tdraft_top1\tint32\t1,15\noutput\t1\tdraft_hidden\tfloat16\t1,15,2048\n";Write(bundle/"step.tsv",wire);auto plan=Observe(bundle/"step.tsv",wire.size());pins.push_back(plan);pins.push_back(Observe(bundle/"draft-abi.json",abi.size(),c.files.at("draft_abi").hash));
    std::vector<std::string> args={c.files.at("draft_runner").path.string(),"--draft-step",(bundle/"step.tsv").string(),"--output",(call/"results").string(),"--expected-runtime-version",c.strings.at("runtime_version"),"--ddk-id",c.strings.at("ddk_id"),"--firmware-id",c.strings.at("firmware_id"),"--timeout-ms",std::to_string(c.timeout),"--model-load-api","legacy-buffer"};
    stage="draft-child";Progress();auto child=Run(args,call,c.strings.at("draft_library_path"),call,progress,children,out);++drafts;draft_child_ms+=child.wall_ms;KeepLogs(call);Require(child.exitcode==0,"draft child failed; no retry");for(const auto& f:pins){Stable(f);Require(Observe(f.path,f.bytes).hash==f.hash,"draft changed input or step bytes");}Pins();
    auto resultbytes=Stat(call/"results/result.json").size;Require(resultbytes<=1<<20,"draft receipt exceeds budget");auto observed_result=ReadObserved(call/"results/result.json",resultbytes);auto resultfile=observed_result.file;std::string text(observed_result.bytes.begin(),observed_result.bytes.end());J native=Parser{text}.parse();
    Require(native.at("schema_version").num()==1&&native.at("state_policy").str()=="prefix-recompute"&&native.at("status").str()=="EXECUTED"&&native.at("scope").str()=="standalone-draft-step"&&native.at("evidence_kind").str()==kEvidence&&native.at("process_id").num()==child.pid&&native.at("target_calls").num()==0&&native.at("draft_calls").num()==1,"draft receipt execution identity mismatch");
    Require(native.at("model_sha256").str()==model.hash&&native.at("draft_abi_sha256").str()==c.files.at("draft_abi").hash&&native.at("step_plan_sha256").str()==plan.hash&&native.at("runtime_version").str()==c.strings.at("runtime_version")&&native.at("ddk_id").str()==c.strings.at("ddk_id")&&native.at("firmware_id").str()==c.strings.at("firmware_id")&&native.at("model_load_api").str()=="legacy-buffer"&&native.at("sequence_length").num()==64,"draft receipt model/ABI/environment mismatch");
    const auto& ni=native.at("inputs");const auto& no=native.at("outputs");Require(ni.kind==J::Array&&ni.a.size()==3&&no.kind==J::Array&&no.a.size()==2,"draft receipt incomplete IO");
    for(std::size_t i=0;i<3;++i){const auto& r=ni.a[i];Require(r.at("index").num()==i&&r.at("name").str()==names[i]&&r.at("dtype").str()==types[i]&&Shape(r.at("shape"),i==0?std::vector<std::uint64_t>{1,64,16384}:std::vector<std::uint64_t>{1,64})&&r.at("bytes").num()==data[i].size()&&r.at("path").str()==pins[i].path.filename().string()&&r.at("sha256").str()==pins[i].hash,"draft receipt input differs from actual prefix bytes");}
    const auto& diag=native.at("io_diagnostic");Require(diag.at("enabled").kind==J::Bool&&diag.at("enabled").b&&diag.at("process_called").kind==J::Bool&&diag.at("process_called").b,"draft native input readback missing");Require(diag.at("output_fill_byte").num()==0&&diag.at("input_readback_point").str()=="AFTER_ALL_INPUT_COPIES_AND_OUTPUT_FILL_BEFORE_PROCESS","draft native readback point/fill mismatch");const auto& outputdiag=diag.at("outputs");Require(outputdiag.kind==J::Array&&outputdiag.a.size()==2,"draft output identity diagnostics incomplete");for(std::size_t i=0;i<2;++i){const auto& r=outputdiag.a[i];Require(r.at("index").num()==i&&r.at("checked_after_process").kind==J::Bool&&r.at("checked_after_process").b&&r.at("object_same").kind==J::Bool&&r.at("object_same").b&&r.at("buffer_same").kind==J::Bool&&r.at("buffer_same").b,"draft output buffer identity changed");}const auto& readback=diag.at("inputs");Require(readback.kind==J::Array&&readback.a.size()==3,"draft readback incomplete");
    for(std::size_t i=0;i<3;++i){auto& r=readback.a[i];Require(r.at("index").num()==i&&r.at("logical_bytes").num()==pins[i].bytes&&r.at("readback_sha256").str()==pins[i].hash&&r.at("readback_matches_source").kind==J::Bool&&r.at("readback_matches_source").b,"draft actual copied input differs");}
    std::vector<File> actual;std::array<Bytes,2> outputbytes;for(std::size_t i=0;i<2;++i){const auto& r=no.a[i];auto name=Numbered("output_",i)+".bin";auto size=i==0?60ULL:61440ULL;Require(r.at("index").num()==i&&r.at("name").str()==(i==0?"draft_top1":"draft_hidden")&&r.at("dtype").str()==(i==0?"int32":"float16")&&Shape(r.at("shape"),i==0?std::vector<std::uint64_t>{1,15}:std::vector<std::uint64_t>{1,15,2048})&&r.at("path").str()==name&&r.at("bytes").num()==size,"draft output ABI differs");HashSyntax(r.at("sha256").str());auto observed=ReadObserved(call/"results"/name,size,r.at("sha256").str());actual.push_back(observed.file);outputbytes[i]=std::move(observed.bytes);Keep(actual.back());}
    const auto& bytes=outputbytes[0];FiniteHalf(outputbytes[1]);std::vector<std::uint32_t> proposals;for(std::size_t i=0;i<15;++i){std::uint32_t token=0;for(int j=0;j<4;++j)token|=std::uint32_t(bytes[4*i+j])<<(8*j);Require(token<kVocab,"draft token outside full vocabulary");proposals.push_back(token);}
    if(c.dump){J receipt=J::Obj();receipt["schema_version"]=1;receipt["artifact_kind"]="qwen35-board-serial-draft-step";receipt["evidence_kind"]=kEvidence;receipt["pid"]=child.pid;receipt["exitcode"]=0;receipt["model_sha256"]=model.hash;receipt["prefix"]=Array(prefix);receipt["feature_prefix_sha256"]=Hash(features);receipt["inputs"]=inputrows;receipt["native_receipt"]=Record(resultfile,out);receipt["proposals"]=Array(proposals);Write(call/"receipt.json",Dump(receipt)+"\n");auto receiptfile=Observe(call/"receipt.json",Stat(call/"receipt.json").size);Keep(receiptfile);Keep(resultfile);Keep(plan);J event=J::Obj();event["receipt"]=Record(receiptfile,out);event["pid"]=child.pid;event["proposals"]=Array(proposals);loop["draft_steps"].a.push_back(event);}
    for(const auto& f:actual){Stable(f);}for(const auto& f:pins){Stable(f);}Stable(resultfile);Pins();if(!c.dump)PruneDraft(call,linked);return proposals;
#endif
  }
  void TextResult(){
#ifdef QWEN35_TEXT_TOKENIZER
    if(!c.tokenizer)return;auto start=Clock::now();
    loop["generated_text"]=c.tokenizer->Decode(generated,true);
    loop["generated_text_with_special_tokens"]=c.tokenizer->Decode(generated,false);
    J info=J::Obj();info["execution"]="BOARD_LOCAL_ENCODE_AND_DECODE";info["prompt_format"]=c.strings.at("prompt_format");
    info["tokenizer_sha256"]=c.files.at("tokenizer_bundle").hash;info["prompt_sha256"]=c.files.at("prompt_text").hash;
    info["prompt_token_count"]=c.prompt.size();info["host_admission_ids_matched"]=true;
    info["unmapped_generated_token_ids"]=Array(c.tokenizer->Unmapped(generated));
    info["invalid_utf8_policy"]="REPLACE_INVALID_OR_INCOMPLETE_SEQUENCES_WITH_U_FFFD";
    info["unmapped_token_policy"]="U_FFFD_WITH_EXPLICIT_ID_LIST";loop["text_io"]=info;
    J timing=J::Obj();timing["load_and_encode_us"]=c.text_setup_us;
    timing["decode_us"]=static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-start).count());
    timing["scope"]="BOARD_CPU_TOKENIZER_WALL";loop["text_latency"]=timing;
#endif
  }
  void Commit(std::uint32_t token,const std::string& role){Target(token,role);generated.push_back(token);if(c.dump){J event=J::Obj();event["token_id"]=token;event["position"]=prefix.size()-1;event["role"]=role;event["target_step_index"]=targets-1;loop["commits"].a.push_back(event);}Progress();}
 public:
  Supervisor(Config config,fs::path directory):c(std::move(config)),out(std::move(directory)){Safe(out,true);if(c.fast){weight_mapping=std::make_unique<MappedTable>(c.files.at("weight_table"));scale_mapping=std::make_unique<MappedTable>(c.files.at("scale_table"));memory_states.resize(89);features.reserve(64*kFeatureBytes);}else{NewDir(out/"target");NewDir(out/"draft");if(c.prefill)NewDir(out/"prefill");}loop["schema_version"]=1;loop["artifact_kind"]="qwen35-board-local-serial-dflash";loop["evidence_kind"]=kEvidence;loop["status"]="RUNNING";loop["device_execution"]="ATTEMPTED";loop["model_compilation"]="NOT_RUN";loop["numerical_validation"]="NOT_RUN";loop["performance"]="NOT_EVALUATED";loop["state_policy"]="ADVANCE_ONLY_COMMITTED_TOKENS_NO_SPECULATIVE_STATE";loop["control_semantics"]="EXPERIMENTAL_SINGLE_QUERY_ABSOLUTE_POSITION";loop["target_steps"]=J::Arr();loop["draft_steps"]=J::Arr();loop["rounds"]=J::Arr();loop["commits"]=J::Arr();loop["models"]=J::Obj();loop["models"]["target_sha256"]=c.files.at("target_model").hash;loop["models"]["draft_sha256"]=c.files.at("draft_model").hash;loop["config_sha256"]=c.config.hash;loop["target_model_sha256"]=c.files.at("target_model").hash;loop["draft_model_sha256"]=c.files.at("draft_model").hash;loop["draft_abi_sha256"]=c.files.at("draft_abi").hash;loop["runtime_version"]=c.strings.at("runtime_version");loop["ddk_id"]=c.strings.at("ddk_id");loop["firmware_id"]=c.strings.at("firmware_id");loop["process_id"]=static_cast<std::uint64_t>(::getpid());progress["status"]="RUNNING";progress["supervisor_pid"]=static_cast<std::uint64_t>(::getpid());progress["launched_child_count"]=0;progress["completed_child_count"]=0;progress["active_child_count"]=0;progress["last_child_pid"]=J();progress["last_child_exitcode"]=J();progress["child_status"]="NOT_STARTED";if(!c.dump){loop["schema_version"]=2;loop["dump"]=false;loop["retrieval_mode"]="FINAL_TOKENS_ONLY";for(auto key:{"target_steps","draft_steps","rounds","commits"})loop.o.erase(key);}if(c.prefill){loop["prompt_ingestion"]="PREFILL64_PADDED_SHORT_PROMPT_EXPERIMENTAL";loop["prefill_model_sha256"]=c.files.at("prefill_model").hash;loop["models"]["prefill_sha256"]=c.files.at("prefill_model").hash;loop["control_semantics"]="EXPERIMENTAL_PADDED_PREFILL_VALID_LENGTH_NOT_NUMERICALLY_VALIDATED";if(c.dump)loop["prefill_steps"]=J::Arr();}
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
    const auto seed=c.config.hash+":"+std::to_string(::getpid());session_id=Sha256(seed);progress["resident_call_status"]="NOT_STARTED";
#ifdef QWEN35_RESIDENT_MIXED
    progress["child_status"]="NONE";prefill_lifecycle.weight_cleanup_policy="MANAGER_DEINIT_ONLY_NO_SEPARATE_WEIGHT_UNLOAD";
#elif defined(QWEN35_RESIDENT_V1)
    progress["child_status"]="NONE";prefill_lifecycle.model_name="qwen35_resident_prefill";prefill_lifecycle.model_load_api="legacy-file";
#endif
#endif
    Progress();}
  int Execute(){try{if(c.prefill)Prefill();else for(auto token:c.prompt)Target(token,"prompt");
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
    StartResident();
#endif
    Commit(next,"initial_target_next_anchor");
    while(generated.size()<c.maxnew){auto proposals=Draft();J round=J::Obj();if(c.dump){round["round"]=round_count;round["prefix_before"]=Array(prefix);round["anchor_token_id"]=prefix.back();round["anchor_position"]=prefix.size()-1;round["proposals"]=Array(proposals);round["committed"]=J::Arr();}std::size_t limit=std::min<std::size_t>(c.k,c.maxnew-generated.size()),used=0,matched=0;bool mismatch=false;
      for(std::size_t i=0;i<limit;++i){auto expected=next;bool match=proposals[i]==expected;++used;++considered;if(match){++accepted;++matched;}else{++corrections;mismatch=true;if(c.dump){round["mismatch_index"]=i;round["correction_token_id"]=expected;}}Commit(expected,match?"accepted_proposal":"target_correction");if(c.dump)round["committed"].a.emplace_back(static_cast<std::uint64_t>(expected));if(!match)break;}
      if(!mismatch&&used==limit&&generated.size()<c.maxnew){auto bonus=next;++bonuses;Commit(bonus,"all_considered_accepted_bonus");if(c.dump){round["bonus_token_id"]=bonus;round["committed"].a.emplace_back(static_cast<std::uint64_t>(bonus));}}
      if(c.dump){round["considered"]=used;round["accepted"]=matched;round["discarded_unverified"]=15-used;round["prefix_after"]=Array(prefix);loop["rounds"].a.push_back(round);}++round_count;stage="round-complete";Progress();}
    Require(round_count>=c.minrounds,"minimum draft rounds not observed");Require(targets==(c.prefill?c.maxnew:c.prompt.size()+c.maxnew)&&generated.size()==c.maxnew&&(!c.prefill||prefills==1),"native target/prefill commit count mismatch");Pins();
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
    stage="resident-close";Progress();Lifecycle("resident-close-start","session="+session_id);session.Close();MemorySnapshot("after-close");Require(session.evidence().closed&&!session.evidence().poisoned,"resident session did not close successfully");Pins();Lifecycle("resident-close-end","session="+session_id);
#endif
    if(c.dump){Write(out/"final-features.bin",features);auto final=Observe(out/"final-features.bin",features.size(),Hash(features));Keep(final);loop["final_features"]=Record(final,out);loop["final_features"]["dtype"]="float16";loop["final_features"]["shape"]=Array({1,static_cast<std::uint32_t>(prefix.size()),16384});loop["final_target_outputs"]=J::Arr();for(std::size_t i=0;i<outputs.size();++i){const auto& f=outputs[i];Stable(f);Keep(f);J row=Record(f,out);row["index"]=i;row["runtime_name"]=c.outputs[i].name;row["tool_type"]=c.outputs[i].type;loop["final_target_outputs"].a.push_back(row);}loop["final_state_files"]=J::Arr();for(std::size_t i=0;i<states.size();++i){Stable(states[i]);J r=Record(states[i],out);r["source_refdata"]=c.states[i].ref;r["runtime_input_index"]=c.states[i].in;r["runtime_output_index"]=c.states[i].out;loop["final_state_files"].a.push_back(r);}}else if(!c.fast){for(const auto& f:outputs)Stable(f);for(const auto& f:states)Stable(f);PruneOwned(last_target_step);outputs.clear();states.clear();last_target_step.clear();Pins();}
    TextResult();loop["status"]="BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED";loop["device_execution"]=std::string(kEvidence)=="device-hiai"?(c.dump?"EXECUTED_BOARD_LOCAL_OUTPUTS_RETAINED":"EXECUTED_BOARD_LOCAL_FINAL_TOKENS_ONLY"):"HOST_FAKE_EXECUTED";loop["eos_policy"]="DISABLED_FIXED_BUDGET_SMOKE";loop["acceptance_scope"]=c.incremental?"CONSIDERED_PROPOSALS_ONLY":"CONSIDERED_K_ONLY_NOT_ALL15";loop["retrieval_manifest"]="retrieval-manifest.tsv";stage="complete";progress["status"]=loop["status"];Progress();Atomic(out/"loop.json",loop);Keep(out/"loop.json");Keep(out/"progress.json");Manifest();return 0;
  }catch(const std::exception& e){
#if defined(QWEN35_RESIDENT_V2) || defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
#if defined(QWEN35_RESIDENT_V1) || defined(QWEN35_RESIDENT_MIXED)
    try{ClosePrefill();}catch(const std::exception& cleanup){loop["prefill_cleanup_error"]=cleanup.what();}
#endif
    try{session.Close();}catch(const std::exception& cleanup){loop["resident_cleanup_error"]=cleanup.what();}
    MemorySnapshot("after-failure-close");progress["resident_call_status"]=session.evidence().closed?"FAILED_CLOSED":"UNRESOLVED";
#endif
    TextResult();loop["status"]="FAILED";loop["error"]=e.what();loop["stage"]=stage;loop["failed_child_pid"]=progress.o.count("last_child_pid")?progress.at("last_child_pid"):J();loop["failed_child_exitcode"]=progress.o.count("last_child_exitcode")?progress.at("last_child_exitcode"):J();progress["status"]="FAILED";Counts();try{Progress();Atomic(out/"loop.json",loop);Keep(out/"loop.json");Keep(out/"progress.json");Manifest();}catch(const std::exception& save){std::cerr<<"failure evidence incomplete: "<<save.what()<<'\n';}std::cerr<<e.what()<<'\n';return 1;}}
};
} // namespace
int main(int argc,char** argv){
  fs::path owned_output;
  ::signal(SIGHUP,SIG_IGN);::signal(SIGPIPE,SIG_IGN);
  try {
    Require(argc==5,"usage: qwen35_board_serial --config ABS.tsv --output NEW_ABS_DIR");
    std::map<std::string,std::string> args;
    for(int i=1;i<argc;i+=2)Require(args.emplace(argv[i],argv[i+1]).second,"duplicate CLI option");
    Require(args.size()==2&&args.count("--config")&&args.count("--output"),"unknown CLI option");
    auto config_path=Absolute(args.at("--config")); auto output=Absolute(args.at("--output"));
    NewDir(output);owned_output=output;
    J bootstrap=J::Obj();bootstrap["status"]="RUNNING";bootstrap["stage"]="validate-config-and-source-hashes";
    bootstrap["supervisor_pid"]=static_cast<std::uint64_t>(::getpid());bootstrap["evidence_kind"]=kEvidence;
    Atomic(output/"progress.json",bootstrap);
    auto config=Load(config_path);Supervisor run(std::move(config),output);return run.Execute();
  }catch(const std::exception& e){
    if(!owned_output.empty())try {
      J failed=J::Obj();failed["schema_version"]=1;failed["artifact_kind"]="qwen35-board-local-serial-dflash";
      failed["status"]="FAILED";failed["stage"]="validate-config-and-source-hashes";failed["error"]=e.what();
      failed["evidence_kind"]=kEvidence;failed["process_id"]=static_cast<std::uint64_t>(::getpid());
      failed["model_compilation"]="NOT_RUN";failed["numerical_validation"]="NOT_RUN";failed["performance"]="NOT_EVALUATED";
      Atomic(owned_output/"progress.json",failed);Atomic(owned_output/"loop.json",failed);
      std::string manifest="QWEN35_BOARD_SERIAL_RETRIEVAL\t1\n";
      for(auto name:{"loop.json","progress.json"}){auto file=Observe(owned_output/name,Stat(owned_output/name).size);manifest+="file\t"+std::string(name)+"\t"+file.hash+"\t"+std::to_string(file.bytes)+"\n";}
      Write(owned_output/"retrieval-manifest.tsv",manifest);
    }catch(const std::exception& save){std::cerr<<"bootstrap failure evidence incomplete: "<<save.what()<<'\n';}
    std::cerr<<e.what()<<'\n';return 1;
  }
}
