// Board-local original40 / full6 serial DFlash execution smoke.
// No network, model compilation, speculative state, or numerical acceptance.
#include "qwen35_dflash/sha256.hpp"
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
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
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
struct File {fs::path path;std::string hash;std::uint64_t bytes;Stamp stamp;};
File Observe(const fs::path& p,std::uint64_t expected,const std::string& hash="") {auto before=Stat(p);Require(before.size==expected,"file extent mismatch: "+p.string());
  auto actual=Sha256File(p);Require(Stat(p)==before,"file changed during hash: "+p.string());if(!hash.empty())Require(actual==hash,"file SHA256 mismatch: "+p.string());return {p,actual,expected,before};}
void Stable(const File& f) {Require(Stat(f.path)==f.stamp,"pinned file changed: "+f.path.string());}
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
struct Config {std::map<std::string,File> files;std::map<std::string,std::string> strings;std::vector<Tensor> inputs,outputs,prefill_inputs,prefill_outputs;std::vector<State> states;std::vector<PrefillState> prefill_states;std::vector<std::uint32_t> prompt;std::uint32_t maxnew=0,k=0,minrounds=0,timeout=0;std::uint64_t startup_hash_ms=0;bool dump=true,prefill=false;Clock::time_point started=Clock::now();File config;};
Config Load(const fs::path& path){Config c;auto configsize=Stat(path).size;Require(configsize<=65536,"config exceeds64KiB");auto configdata=ReadObserved(path,configsize);c.config=configdata.file;std::string text(configdata.bytes.begin(),configdata.bytes.end());
  for(unsigned char x:text)Require((x>=32&&x<=126)||x=='\n'||x=='\t',"TSV must be ASCII LF");
  auto lines=Split(text,'\n');Require(lines.back().empty(),"config needs final LF");lines.pop_back();
  Require(!lines.empty()&&lines[0]=="QWEN35_BOARD_SERIAL\t1","config header mismatch");
  const std::set<std::string> filekeys={"target_model","prefill_model","draft_model","draft_abi","target_tool","draft_runner","weight_table","scale_table"};
  const std::set<std::string> strkeys={"target_library_path","draft_library_path","runtime_version","ddk_id","firmware_id","prompt","max_new_tokens","max_proposals","min_draft_rounds","timeout_ms"};
  for(std::size_t n=1;n<lines.size();++n){auto f=Split(lines[n],'\t');Require(!f[0].empty(),"empty config row");
    if(filekeys.count(f[0])){Require(f.size()==4&&!c.files.count(f[0]),"file row malformed/duplicate");HashSyntax(f[2]);auto bytes=Integer(f[3],32ULL<<30,false);auto p=Absolute(f[1]);
      if(f[0]=="weight_table")Require(bytes==kVocab*2048ULL,"weight table extent mismatch");
      else if(f[0]=="scale_table")Require(bytes==kVocab*4ULL,"scale table extent mismatch");
      else if(f[0]=="draft_model")Require(bytes<=UINT32_MAX,"draft model exceeds legacy32-bit extent");
      else if(f[0]=="draft_abi")Require(bytes<=1ULL<<20,"draft ABI exceeds budget");
      else Require(bytes<=64ULL<<20,"model/tool exceeds budget");
      // Full table/model hashes occur once at initialization, never per token.
      Lifecycle("startup-hash-begin","role="+f[0]+" bytes="+std::to_string(bytes));
      auto hash_begin=Clock::now();c.files.emplace(f[0],Observe(p,bytes,f[2]));
      auto hash_ms=ElapsedMs(hash_begin);c.startup_hash_ms+=hash_ms;
      Lifecycle("startup-hash-end","role="+f[0]+" wall_ms="+std::to_string(hash_ms));}
    else if(f[0]=="dump"){Require(f.size()==2&&!c.strings.count("dump")&&(f[1]=="0"||f[1]=="1"),"dump must be one explicit0/1 singleton");c.dump=f[1]=="1";c.strings.emplace("dump",f[1]);}
    else if(strkeys.count(f[0])){Require(f.size()==2&&!f[1].empty()&&!c.strings.count(f[0]),"singleton row malformed/duplicate");c.strings.emplace(f[0],f[1]);}
    else if(f[0]=="target_input"||f[0]=="target_output"||f[0]=="prefill_input"||f[0]=="prefill_output"){Require(f.size()==5,"tensor row malformed");auto& v=f[0]=="target_input"?c.inputs:f[0]=="target_output"?c.outputs:f[0]=="prefill_input"?c.prefill_inputs:c.prefill_outputs;
      auto index=Integer(f[1],100);Require(index==v.size(),"tensor indices must be ordered/contiguous");if(f[0]=="target_input"||f[0]=="prefill_input")Name(f[2]);else Require(!f[2].empty()&&f[2].size()<=256,"runtime output name exceeds budget");Require(std::set<std::string>{"FP16","FP32","INT8","INT16","INT32","INT64"}.count(f[3]),"tooltype invalid");
      v.push_back({static_cast<std::uint32_t>(index),f[2],f[3],Integer(f[4],8ULL<<20,false)});}
    else if(f[0]=="state"){Require(f.size()==5,"state row malformed");Name(f[1]);c.states.push_back({f[1],static_cast<std::uint32_t>(Integer(f[2],88)),static_cast<std::uint32_t>(Integer(f[3],81)),Integer(f[4],8ULL<<20,false)});}
    else if(f[0]=="prefill_state"){Require(f.size()==6,"prefill state row malformed");Name(f[1]);Name(f[2]);c.prefill_states.push_back({f[1],f[2],static_cast<std::uint32_t>(Integer(f[3],88)),static_cast<std::uint32_t>(Integer(f[4],81)),Integer(f[5],8ULL<<20,false)});}
    else Fail("unknown config row: "+f[0]);}
  c.prefill=c.files.count("prefill_model")!=0;
  Require(c.files.size()==filekeys.size()-1+static_cast<std::size_t>(c.prefill)&&c.strings.size()==strkeys.size()+c.strings.count("dump"),"config fields missing");
  Require(c.prefill?(c.prefill_inputs.size()==89&&c.prefill_outputs.size()==82&&c.prefill_states.size()==80):(c.prefill_inputs.empty()&&c.prefill_outputs.empty()&&c.prefill_states.empty()),"prefill config must be complete89/82/80 or entirely absent");
  Require(c.files.at("weight_table").bytes==kVocab*2048ULL&&c.files.at("scale_table").bytes==kVocab*4ULL,"original table extents mismatch");
  Require(c.files.at("target_model").bytes<=64ULL<<20&&c.files.at("draft_model").bytes<=UINT32_MAX&&c.files.at("draft_abi").bytes<=1<<20,"model/ABI budget mismatch");
  for(auto key:{"target_library_path","draft_library_path"})for(const auto& s:Split(c.strings.at(key),':'))Safe(Absolute(s),true);
  for(auto key:{"target_tool","draft_runner"})Require(::access(c.files.at(key).path.c_str(),X_OK)==0,"child executable is not executable");
  for(const auto& s:Split(c.strings.at("prompt"),','))c.prompt.push_back(static_cast<std::uint32_t>(Integer(s,kVocab-1)));
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
class Supervisor {
  Config c;fs::path out,last_target_step;J loop=J::Obj(),progress=J::Obj(),children=J::Arr();std::vector<File> sources,inputs,outputs,states;std::vector<std::uint32_t> prefix,generated;Bytes features;std::uint32_t next=0;std::set<std::string> retrieve;std::map<std::string,File> kept;std::size_t targets=0,drafts=0,prefills=0,prompt_calls=0,round_count=0,considered=0,accepted=0,corrections=0,bonuses=0;std::uint64_t target_child_ms=0,draft_child_ms=0,prompt_child_ms=0,prefill_child_ms=0;std::map<std::uint64_t,std::string> zero_hashes;std::string stage="initialize";
  void Pins(){for(const auto& x:c.files)Stable(x.second);Stable(c.config);for(const auto& x:kept)Stable(x.second);}
  void Keep(const File& f){
    if(!c.dump&&f.path!=out/"loop.json"&&f.path!=out/"progress.json")return;
    Stable(f);auto rel=fs::relative(f.path,out).generic_string();
    if(kept.count(rel)){const auto& old=kept.at(rel);Require(old.stamp==f.stamp&&old.hash==f.hash&&old.bytes==f.bytes,"retained observation differs: "+f.path.string());Stable(old);return;}
    kept.emplace(rel,f);retrieve.insert(rel);
  }
  void Keep(const fs::path& p){if(!c.dump&&p!=out/"loop.json"&&p!=out/"progress.json")return;auto rel=fs::relative(p,out).generic_string();if(kept.count(rel)){Stable(kept.at(rel));return;}Keep(Observe(p,Stat(p).size));}
  void KeepLogs(const fs::path& p){if(!c.dump)return;for(auto name:{"tool.stdout","tool.stderr","tool.pid","tool.exitcode","command.json"})if(fs::exists(p/name))Keep(p/name);}
  void Counts(){J n=J::Obj();n["target_native_calls"]=targets;n["draft_native_calls"]=drafts;if(c.prefill)n["prefill_native_calls"]=prefills;n["considered"]=considered;n["accepted"]=accepted;n["corrections"]=corrections;n["bonuses"]=bonuses;loop["counts"]=n;loop["prompt"]=Array(c.prompt);loop["generated_tokens"]=Array(generated);loop["full_prefix"]=Array(prefix);loop["prompt_token_ids"]=Array(c.prompt);loop["generated_token_ids"]=Array(generated);loop["full_prefix_token_ids"]=Array(prefix);J timing=J::Obj();timing["scope"]="MONOTONIC_WALL_CLOCK_NOT_NPU_INFERENCE";timing["startup_source_hash_ms"]=c.startup_hash_ms;timing["target_child_process_wall_ms"]=target_child_ms;timing["draft_child_process_wall_ms"]=draft_child_ms;timing["target_child_processes_timed"]=targets;timing["draft_child_processes_timed"]=drafts;timing["total_wall_ms"]=ElapsedMs(c.started);if(c.prefill){timing["prefill_child_process_wall_ms"]=prefill_child_ms;timing["prefill_child_processes_timed"]=prefills;}loop["supervisor_timing"]=timing;J acceptance=J::Obj();acceptance["accepted"]=accepted;acceptance["considered"]=considered;acceptance["rate"]=Ratio(accepted,considered);loop["acceptance"]=acceptance;J phases=J::Obj();phases["timing_scope"]="CHILD_WALL_INCLUDES_LOAD_RUN_UNLOAD";phases["inference_only_status"]="NOT_PROVIDED";auto phase=[](std::uint64_t calls,std::uint64_t ms){J row=J::Obj();row["calls"]=calls;row["total_ms"]=ms;row["mean_ms"]=Ratio(ms,calls);return row;};phases["prompt_ingest"]=phase(c.prefill?prefills:prompt_calls,c.prefill?prefill_child_ms:prompt_child_ms);phases["prompt_ingest"]["mechanism"]=c.prefill?"PREFILL64_PADDED_SHORT_PROMPT_EXPERIMENTAL":"DECODE1_SERIAL_PROMPT_INGESTION";if(c.prefill){phases["prefill"]=phase(prefills,prefill_child_ms);phases["prompt_ingest"]["latency_alias"]="prefill";}phases["decode"]=phase(targets-prompt_calls,target_child_ms-prompt_child_ms);phases["draft"]=phase(drafts,draft_child_ms);loop["phase_latency"]=phases;if(!c.dump){loop.o.erase("prompt_token_ids");loop.o.erase("generated_token_ids");loop.o.erase("full_prefix_token_ids");loop["completed_rounds"]=round_count;J life=J::Obj();for(auto key:{"launched_child_count","completed_child_count","active_child_count","last_child_pid","last_child_exitcode","child_status"})life[key]=progress.at(key);life["children"]=children;loop["child_lifecycle"]=life;}}
  void Progress(){Counts();progress["stage"]=stage;progress["committed_prefix_length"]=prefix.size();progress["generated_tokens"]=Array(generated);progress["counts"]=loop["counts"];progress["supervisor_timing"]=loop["supervisor_timing"];Atomic(out/"progress.json",progress);}
  void PruneOwned(const fs::path& tree){
    Require(!c.dump&&progress.at("child_status").str()=="EXITED"&&progress.at("last_child_exitcode").num()==0&&progress.at("active_child_count").num()==0,"scratch cleanup requires a known successful child exit");
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
    auto step=out/"prefill/step_0000";NewDir(step);NewDir(step/"inputs");NewDir(step/"outputs");inputs.clear();
    const auto valid=c.prompt.size();Bytes embeddings(64*2048,0),scales(64*4,0),positions(64*4,0),conv(512,0),mask(64*8192,0);
    for(std::size_t i=0;i<valid;++i){auto row=TableRow(c.files.at("weight_table"),c.prompt[i],2048),scale=TableRow(c.files.at("scale_table"),c.prompt[i],4);std::uint32_t bits=0;for(int n=0;n<4;++n)bits|=std::uint32_t(scale[n])<<(8*n);float value;std::memcpy(&value,&bits,4);Require(std::isfinite(value)&&value>=0,"invalid original prefill embedding scale");std::copy(row.begin(),row.end(),embeddings.begin()+i*2048);std::copy(scale.begin(),scale.end(),scales.begin()+i*4);}
    for(std::size_t q=0;q<64;++q){auto word=Scalar(q,4);std::copy(word.begin(),word.end(),positions.begin()+q*4);for(std::size_t key=0;key<2048;++key){const bool allowed=q<valid?key<=q:key==0;if(!allowed){mask[q*8192+key*4+2]=0x80;mask[q*8192+key*4+3]=0xff;}}}
    auto length=Scalar(valid,4);std::copy(length.begin(),length.end(),conv.begin());
    std::map<std::string,Bytes> controls={{"arg1_1",std::move(embeddings)},{"arg0_1",std::move(scales)},{"arg107_1",positions},{"arg2_1",positions},{"arg1238_1",Scalar(valid-1,4)},{"arg114_1",Scalar(valid,8)},{"arg18_1",Scalar(valid,2)},{"arg14_1",std::move(conv)},{"arg113_1",std::move(mask)}};
    std::set<unsigned> state_inputs;for(const auto& s:c.prefill_states)state_inputs.insert(s.in);
    J inputrows=J::Arr(),outputrows=J::Arr();std::string paths,types,outtypes;
    for(const auto& t:c.prefill_inputs){auto path=step/"inputs"/(Numbered("input_",t.index)+".bin");Bytes data=state_inputs.count(t.index)?Bytes(t.bytes,0):controls.at(t.name);Require(data.size()==t.bytes,"prefill input extent mismatch");Write(path,data);auto f=Observe(path,t.bytes,Hash(data));inputs.push_back(f);J row=Record(f,out);row["index"]=t.index;row["name"]=t.name;row["tool_type"]=t.type;inputrows.a.push_back(row);if(!paths.empty()){paths+=',';types+=',';}paths+=path.string();types+=t.type;}
    for(const auto& t:c.prefill_outputs){if(!outtypes.empty())outtypes+=',';outtypes+=t.type;}
    for(const auto& f:inputs){Stable(f);}Pins();stage="prefill-child";Progress();
    std::vector<std::string> args={c.files.at("target_tool").path.string(),"--model="+c.files.at("prefill_model").path.string(),"--input="+paths,"--times=1","--modelmanager_itf=V2","--weight_dir="+c.files.at("prefill_model").path.parent_path().string(),"--enable_item=1","--direct_from_file=true","--input_type="+types,"--output_type="+outtypes,"--output_dir="+(step/"outputs").string()};
    auto child=Run(args,step,c.strings.at("target_library_path"),step,progress,children,out);++prefills;prefill_child_ms+=child.wall_ms;KeepLogs(step);Require(child.exitcode==0,"prefill child failed; no retry");
    for(const auto& f:inputs){Stable(f);Require(Observe(f.path,f.bytes).hash==f.hash,"prefill changed private input bytes");}Pins();stage="prefill-bind-outputs";
    auto log=Text(step/"tool.stdout",1<<20);std::map<unsigned,fs::path> mapping;std::istringstream ls(log);std::string line;std::regex pattern("^output\\[([0-9]+)\\][ \\t]*:[ \\t]*(/[^ \\t\\r]+)[ \\t\\r]*$");
    while(std::getline(ls,line)){std::smatch m;if(std::regex_match(line,m,pattern)){auto index=Integer(m[1],81);Require(!mapping.count(index),"duplicate logged prefill output");auto p=Absolute(m[2]);Require(p.parent_path()==step/"outputs","prefill output outside owned attempt");mapping.emplace(index,p);}else if(line.find("output[")!=std::string::npos)Fail("malformed prefill output log mapping");}
    Require(mapping.size()==82&&log.find("Execute result :Success")!=std::string::npos,"prefill log missing complete explicit output mapping/success");
    std::set<fs::path> actual,selected;for(const auto& e:fs::directory_iterator(step/"outputs")){Safe(e.path());actual.insert(e.path());}Require(actual.size()==82,"prefill output directory count differs");outputs.clear();Bytes logits,feature_rows;
    for(const auto& t:c.prefill_outputs){auto path=mapping.at(t.index);Require(actual.count(path)&&selected.insert(path).second,"duplicate/absent prefill output file");File f;if(t.index==0||t.index==81){auto observed=ReadObserved(path,t.bytes);f=observed.file;(t.index==0?logits:feature_rows)=std::move(observed.bytes);}else f=Observe(path,t.bytes);outputs.push_back(f);J row=Record(f,out);row["index"]=t.index;row["runtime_name"]=t.name;row["tool_type"]=t.type;outputrows.a.push_back(row);Keep(f);}
    next=Argmax(logits);Require(feature_rows.size()==64*kFeatureBytes,"prefill feature physical64 extent mismatch");feature_rows.resize(valid*kFeatureBytes);FiniteHalf(feature_rows);
    std::map<std::string,PrefillState> bridge;for(const auto& s:c.prefill_states)bridge.emplace(s.decode_ref,s);J state_rows=J::Arr();std::vector<File> newstates;
    for(const auto& decode:c.states){const auto& s=bridge.at(decode.ref);const auto& f=outputs.at(s.out);Stable(f);Require(f.bytes==decode.bytes,"prefill state handoff byte extent differs");newstates.push_back(f);J row=J::Obj();row["decode_source_refdata"]=decode.ref;row["decode_runtime_input_index"]=decode.in;row["decode_runtime_output_index"]=decode.out;row["prefill_source_refdata"]=s.ref;row["prefill_runtime_input_index"]=s.in;row["prefill_runtime_output_index"]=s.out;row["bytes"]=s.bytes;row["prefill_output"]=Record(f,out);row["initialization"]="ZERO_FP16";state_rows.a.push_back(row);}
    if(c.dump){J receipt=J::Obj();receipt["schema_version"]=1;receipt["artifact_kind"]="qwen35-board-serial-prefill-step";receipt["evidence_kind"]=kEvidence;receipt["status"]="EXECUTED_OUTPUTS_BOUND";receipt["pid"]=child.pid;receipt["exitcode"]=0;receipt["model_sha256"]=c.files.at("prefill_model").hash;receipt["prompt_token_ids"]=Array(c.prompt);receipt["logical_prompt_length"]=valid;receipt["physical_sequence_length"]=64;receipt["selected_output_pos"]=valid-1;receipt["control_semantics"]="EXPERIMENTAL_PADDED_PREFILL_VALID_LENGTH_NOT_NUMERICALLY_VALIDATED";receipt["inputs"]=inputrows;receipt["outputs"]=outputrows;receipt["state_mapping"]=state_rows;receipt["next_token_id"]=next;J feature=J::Obj();feature["source"]=Record(outputs.at(81),out);feature["row_start"]=0;feature["row_count"]=valid;feature["bytes"]=feature_rows.size();feature["sha256"]=Hash(feature_rows);receipt["feature_prefix"]=feature;Write(step/"receipt.json",Dump(receipt)+"\n");auto file=Observe(step/"receipt.json",Stat(step/"receipt.json").size);Keep(file);J event=J::Obj();event["receipt"]=Record(file,out);event["pid"]=child.pid;event["prompt_length"]=valid;event["next_token_id"]=next;loop["prefill_steps"].a.push_back(event);}
    for(const auto& f:outputs){Stable(f);}for(const auto& f:inputs){Stable(f);}Pins();states=std::move(newstates);features=std::move(feature_rows);prefix=c.prompt;
    if(!c.dump){PruneOwned(step/"inputs");inputs.clear();last_target_step=step;}stage="prefill-committed";Progress();
  }
  void Target(std::uint32_t token,const std::string& role){stage="target-inputs";Pins();for(const auto& f:states)Stable(f);auto step=out/"target"/Numbered("step_",targets);NewDir(step);NewDir(step/"inputs");NewDir(step/"outputs");auto oldstates=states;inputs.clear();
    Bytes row=TableRow(c.files.at("weight_table"),token,2048),scale=TableRow(c.files.at("scale_table"),token,4);std::uint32_t sb=0;for(int i=0;i<4;++i)sb|=std::uint32_t(scale[i])<<(8*i);float sf;std::memcpy(&sf,&sb,4);Require(std::isfinite(sf)&&sf>=0,"invalid original embedding scale");
    std::map<std::string,Bytes> control={{"arg1_1",row},{"arg0_1",scale},{"arg107_1",Scalar(prefix.size(),4)},{"arg2_1",Scalar(prefix.size(),4)},{"arg1238_1",Scalar(0,4)},{"arg114_1",Scalar(1,8)},{"arg18_1",Scalar(1,2)},{"arg14_1",Bytes(512,0)},{"arg113_1",Bytes(8192,0)}};
    control.at("arg14_1")[0]=1;for(std::size_t i=prefix.size()+1;i<2048;++i){control.at("arg113_1")[i*4+2]=0x80;control.at("arg113_1")[i*4+3]=0xff;}
    std::map<unsigned,std::size_t> byinput;for(std::size_t i=0;i<c.states.size();++i)byinput[c.states[i].in]=i;
    J inputrows=J::Arr(),feedback=J::Arr();std::string paths,types,outtypes;
    for(const auto& t:c.inputs){auto path=step/"inputs"/(Numbered("input_",t.index)+".bin");Bytes b;auto found=byinput.find(t.index);if(found==byinput.end())b=control.at(t.name);else if(oldstates.empty())b=Bytes(t.bytes,0);else b=Read(oldstates[found->second].path,t.bytes);
      Require(b.size()==t.bytes,"target input extent mismatch");
      std::string expected_hash;
      if(found==byinput.end())expected_hash=Hash(b);
      else if(!oldstates.empty()){Stable(oldstates[found->second]);expected_hash=oldstates[found->second].hash;}
      else {auto known=zero_hashes.find(t.bytes);if(known==zero_hashes.end())known=zero_hashes.emplace(t.bytes,Hash(b)).first;expected_hash=known->second;}
      Write(path,b);auto f=Observe(path,t.bytes,expected_hash);inputs.push_back(f);J r=Record(f,out);r["index"]=t.index;r["name"]=t.name;r["tool_type"]=t.type;inputrows.a.push_back(r);
      if(found!=byinput.end()){J state=J::Obj();state["source_refdata"]=c.states[found->second].ref;state["runtime_input_index"]=t.index;state["runtime_output_index"]=c.states[found->second].out;state["input"]=r;if(!oldstates.empty()){Stable(oldstates[found->second]);Require(f.hash==oldstates[found->second].hash,"state input differs from committed parent");state["parent_output"]=Record(oldstates[found->second],out);}else state["initialization"]="ZERO_FP16";feedback.a.push_back(state);}
      if(!paths.empty()){paths+=',';types+=',';}paths+=path.string();types+=t.type;}
    for(const auto& t:c.outputs){if(!outtypes.empty())outtypes+=',';outtypes+=t.type;}
    for(const auto& f:inputs)Stable(f);
    Pins();stage="target-child";Progress();
    std::vector<std::string> args={c.files.at("target_tool").path.string(),"--model="+c.files.at("target_model").path.string(),"--input="+paths,"--times=1","--modelmanager_itf=V2","--weight_dir="+c.files.at("target_model").path.parent_path().string(),"--enable_item=1","--direct_from_file=true","--input_type="+types,"--output_type="+outtypes,"--output_dir="+(step/"outputs").string()};
    auto child=Run(args,step,c.strings.at("target_library_path"),step,progress,children,out);++targets;target_child_ms+=child.wall_ms;if(role=="prompt"){++prompt_calls;prompt_child_ms+=child.wall_ms;}KeepLogs(step);Require(child.exitcode==0,"target child failed; no retry");
    for(const auto& f:inputs){Stable(f);Require(Observe(f.path,f.bytes).hash==f.hash,"target changed private input bytes");}for(const auto& f:oldstates)Stable(f);Pins();
    stage="target-bind-outputs";auto log=Text(step/"tool.stdout",1<<20);std::map<unsigned,fs::path> mapping;std::istringstream ls(log);std::string line;
    std::regex pattern("^output\\[([0-9]+)\\][ \\t]*:[ \\t]*(/[^ \\t\\r]+)[ \\t\\r]*$");
    while(std::getline(ls,line)){std::smatch m;if(std::regex_match(line,m,pattern)){auto index=Integer(m[1],81);Require(!mapping.count(index),"duplicate logged target output");auto p=Absolute(m[2]);Require(p.parent_path()==step/"outputs","logged output outside owned attempt");mapping.emplace(index,p);}else if(line.find("output[")!=std::string::npos)Fail("malformed target output log mapping");}
    Require(mapping.size()==82&&log.find("Execute result :Success")!=std::string::npos,"target log missing complete explicit output mapping/success");
    std::set<fs::path> actual;for(const auto& e:fs::directory_iterator(step/"outputs")){Safe(e.path());actual.insert(e.path());}Require(actual.size()==82,"target output directory count differs");outputs.clear();J outputrows=J::Arr();std::set<fs::path> selected;
    Bytes logits,chunk;
    for(const auto& t:c.outputs){auto p=mapping.at(t.index);Require(actual.count(p)&&selected.insert(p).second,"duplicate or absent target file mapping");File f;
      if(t.index==0||t.index==81){auto observed=ReadObserved(p,t.bytes);f=observed.file;(t.index==0?logits:chunk)=std::move(observed.bytes);}else f=Observe(p,t.bytes);
      outputs.push_back(f);J r=Record(f,out);r["index"]=t.index;r["runtime_name"]=t.name;r["tool_type"]=t.type;outputrows.a.push_back(r);}
    next=Argmax(logits);FiniteHalf(chunk);
    if(c.dump){J receipt=J::Obj();receipt["schema_version"]=1;receipt["artifact_kind"]="qwen35-board-serial-target-step";receipt["evidence_kind"]=kEvidence;receipt["status"]="EXECUTED_OUTPUTS_BOUND";receipt["pid"]=child.pid;receipt["exitcode"]=0;receipt["token_id"]=token;receipt["absolute_position"]=prefix.size();receipt["role"]=role;receipt["model_sha256"]=c.files.at("target_model").hash;receipt["inputs"]=inputrows;receipt["outputs"]=outputrows;receipt["state_feedback"]=feedback;receipt["next_token_id"]=next;receipt["control_semantics"]="EXPERIMENTAL_SINGLE_QUERY_ABSOLUTE_POSITION";
    Write(step/"receipt.json",Dump(receipt)+"\n");auto receiptfile=Observe(step/"receipt.json",Stat(step/"receipt.json").size);Keep(receiptfile);J event=J::Obj();event["receipt"]=Record(receiptfile,out);event["pid"]=child.pid;event["token_id"]=token;event["next_token_id"]=next;loop["target_steps"].a.push_back(event);}
    for(const auto& f:outputs){Stable(f);}for(const auto& f:inputs){Stable(f);}Pins();
    states.clear();for(const auto& s:c.states)states.push_back(outputs[s.out]);features.insert(features.end(),chunk.begin(),chunk.end());prefix.push_back(token);
    if(!c.dump){PruneOwned(step/"inputs");inputs.clear();if(!last_target_step.empty())PruneOwned(last_target_step);last_target_step=step;}stage="target-committed";Progress();
  }
  std::vector<std::uint32_t> Draft(){stage="draft-inputs";Pins();Require(!prefix.empty()&&prefix.size()<=64&&features.size()==prefix.size()*kFeatureBytes,"draft prefix/features mismatch");auto call=out/"draft"/Numbered("call-",drafts);NewDir(call);auto bundle=call/"bundle";NewDir(bundle);NewDir(bundle/"model");
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
  }
  void Commit(std::uint32_t token,const std::string& role){Target(token,role);generated.push_back(token);if(c.dump){J event=J::Obj();event["token_id"]=token;event["position"]=prefix.size()-1;event["role"]=role;event["target_step_index"]=targets-1;loop["commits"].a.push_back(event);}Progress();}
 public:
  Supervisor(Config config,fs::path directory):c(std::move(config)),out(std::move(directory)){Safe(out,true);NewDir(out/"target");NewDir(out/"draft");if(c.prefill)NewDir(out/"prefill");loop["schema_version"]=1;loop["artifact_kind"]="qwen35-board-local-serial-dflash";loop["evidence_kind"]=kEvidence;loop["status"]="RUNNING";loop["device_execution"]="ATTEMPTED";loop["model_compilation"]="NOT_RUN";loop["numerical_validation"]="NOT_RUN";loop["performance"]="NOT_EVALUATED";loop["state_policy"]="ADVANCE_ONLY_COMMITTED_TOKENS_NO_SPECULATIVE_STATE";loop["control_semantics"]="EXPERIMENTAL_SINGLE_QUERY_ABSOLUTE_POSITION";loop["target_steps"]=J::Arr();loop["draft_steps"]=J::Arr();loop["rounds"]=J::Arr();loop["commits"]=J::Arr();loop["models"]=J::Obj();loop["models"]["target_sha256"]=c.files.at("target_model").hash;loop["models"]["draft_sha256"]=c.files.at("draft_model").hash;loop["config_sha256"]=c.config.hash;loop["target_model_sha256"]=c.files.at("target_model").hash;loop["draft_model_sha256"]=c.files.at("draft_model").hash;loop["draft_abi_sha256"]=c.files.at("draft_abi").hash;loop["runtime_version"]=c.strings.at("runtime_version");loop["ddk_id"]=c.strings.at("ddk_id");loop["firmware_id"]=c.strings.at("firmware_id");loop["process_id"]=static_cast<std::uint64_t>(::getpid());progress["status"]="RUNNING";progress["supervisor_pid"]=static_cast<std::uint64_t>(::getpid());progress["launched_child_count"]=0;progress["completed_child_count"]=0;progress["active_child_count"]=0;progress["last_child_pid"]=J();progress["last_child_exitcode"]=J();progress["child_status"]="NOT_STARTED";if(!c.dump){loop["schema_version"]=2;loop["dump"]=false;loop["retrieval_mode"]="FINAL_TOKENS_ONLY";for(auto key:{"target_steps","draft_steps","rounds","commits"})loop.o.erase(key);}if(c.prefill){loop["prompt_ingestion"]="PREFILL64_PADDED_SHORT_PROMPT_EXPERIMENTAL";loop["prefill_model_sha256"]=c.files.at("prefill_model").hash;loop["models"]["prefill_sha256"]=c.files.at("prefill_model").hash;loop["control_semantics"]="EXPERIMENTAL_PADDED_PREFILL_VALID_LENGTH_NOT_NUMERICALLY_VALIDATED";if(c.dump)loop["prefill_steps"]=J::Arr();}Progress();}
  int Execute(){try{if(c.prefill)Prefill();else for(auto token:c.prompt)Target(token,"prompt");Commit(next,"initial_target_next_anchor");
    while(generated.size()<c.maxnew){auto proposals=Draft();J round=J::Obj();round["round"]=round_count;round["prefix_before"]=Array(prefix);round["anchor_token_id"]=prefix.back();round["anchor_position"]=prefix.size()-1;round["proposals"]=Array(proposals);round["committed"]=J::Arr();std::size_t limit=std::min<std::size_t>(c.k,c.maxnew-generated.size()),used=0,matched=0;bool mismatch=false;
      for(std::size_t i=0;i<limit;++i){auto expected=next;bool match=proposals[i]==expected;++used;++considered;if(match){++accepted;++matched;}else{++corrections;mismatch=true;round["mismatch_index"]=i;round["correction_token_id"]=expected;}Commit(expected,match?"accepted_proposal":"target_correction");round["committed"].a.emplace_back(static_cast<std::uint64_t>(expected));if(!match)break;}
      if(!mismatch&&used==limit&&generated.size()<c.maxnew){auto bonus=next;++bonuses;Commit(bonus,"all_considered_accepted_bonus");round["bonus_token_id"]=bonus;round["committed"].a.emplace_back(static_cast<std::uint64_t>(bonus));}
      round["considered"]=used;round["accepted"]=matched;round["discarded_unverified"]=15-used;round["prefix_after"]=Array(prefix);if(c.dump)loop["rounds"].a.push_back(round);++round_count;stage="round-complete";Progress();}
    Require(round_count>=c.minrounds,"minimum draft rounds not observed");Require(targets==(c.prefill?c.maxnew:c.prompt.size()+c.maxnew)&&generated.size()==c.maxnew&&(!c.prefill||prefills==1),"native target/prefill commit count mismatch");Pins();
    if(c.dump){Write(out/"final-features.bin",features);auto final=Observe(out/"final-features.bin",features.size(),Hash(features));Keep(final);loop["final_features"]=Record(final,out);loop["final_features"]["dtype"]="float16";loop["final_features"]["shape"]=Array({1,static_cast<std::uint32_t>(prefix.size()),16384});loop["final_target_outputs"]=J::Arr();for(std::size_t i=0;i<outputs.size();++i){const auto& f=outputs[i];Stable(f);Keep(f);J row=Record(f,out);row["index"]=i;row["runtime_name"]=c.outputs[i].name;row["tool_type"]=c.outputs[i].type;loop["final_target_outputs"].a.push_back(row);}loop["final_state_files"]=J::Arr();for(std::size_t i=0;i<states.size();++i){Stable(states[i]);J r=Record(states[i],out);r["source_refdata"]=c.states[i].ref;r["runtime_input_index"]=c.states[i].in;r["runtime_output_index"]=c.states[i].out;loop["final_state_files"].a.push_back(r);}}else{for(const auto& f:outputs)Stable(f);for(const auto& f:states)Stable(f);PruneOwned(last_target_step);outputs.clear();states.clear();last_target_step.clear();Pins();}
    loop["status"]="BOARD_SERIAL_DFLASH_EXECUTED_UNVALIDATED";loop["device_execution"]=std::string(kEvidence)=="device-hiai"?(c.dump?"EXECUTED_BOARD_LOCAL_OUTPUTS_RETAINED":"EXECUTED_BOARD_LOCAL_FINAL_TOKENS_ONLY"):"HOST_FAKE_EXECUTED";loop["eos_policy"]="DISABLED_FIXED_BUDGET_SMOKE";loop["acceptance_scope"]="CONSIDERED_K_ONLY_NOT_ALL15";loop["retrieval_manifest"]="retrieval-manifest.tsv";stage="complete";progress["status"]=loop["status"];Progress();Atomic(out/"loop.json",loop);Keep(out/"loop.json");Keep(out/"progress.json");Manifest();return 0;
  }catch(const std::exception& e){loop["status"]="FAILED";loop["error"]=e.what();loop["stage"]=stage;loop["failed_child_pid"]=progress.o.count("last_child_pid")?progress.at("last_child_pid"):J();loop["failed_child_exitcode"]=progress.o.count("last_child_exitcode")?progress.at("last_child_exitcode"):J();progress["status"]="FAILED";Counts();try{Progress();Atomic(out/"loop.json",loop);Keep(out/"loop.json");Keep(out/"progress.json");Manifest();}catch(const std::exception& save){std::cerr<<"failure evidence incomplete: "<<save.what()<<'\n';}std::cerr<<e.what()<<'\n';return 1;}}
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
