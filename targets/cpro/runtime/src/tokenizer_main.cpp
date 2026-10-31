#include "qwen35_dflash/tokenizer.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

// Local parity utility; this binary needs neither HIAI nor model weights.
// stdin: one hex UTF-8 prompt per line, or comma-separated IDs in decode mode.
int main(int argc,char** argv){try{
  if(argc!=3)throw std::runtime_error("usage: qwen35_tokenizer BUNDLE raw|chat|decode|decode-all");
  qwen35::dflash::TextTokenizer tokenizer(argv[1]);std::string line,mode=argv[2];
  while(std::getline(std::cin,line)){
    if(mode=="decode"||mode=="decode-all"){
      std::vector<std::uint32_t> ids;std::istringstream parts(line);std::string part;
      while(std::getline(parts,part,','))ids.push_back(std::stoul(part));
      auto s=tokenizer.Decode(ids,mode=="decode");const char* h="0123456789abcdef";for(unsigned char c:s)std::cout<<h[c>>4]<<h[c&15];std::cout<<'\n';
    }else{if(line.size()%2)throw std::runtime_error("odd hex prompt");std::string s;for(std::size_t i=0;i<line.size();i+=2)s+=char(std::stoul(line.substr(i,2),nullptr,16));
      auto ids=tokenizer.Encode(s,mode);for(std::size_t i=0;i<ids.size();++i){if(i)std::cout<<',';std::cout<<ids[i];}std::cout<<'\n';}
  }return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
