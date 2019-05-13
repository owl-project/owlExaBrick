// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <stack>
#include "ScalarField.h"

namespace exa {

  ScalarField::SP ScalarField::load(const std::string &fieldName, const std::string &fileName)
  {
    std::ifstream scalarFile(fileName);
    if (!scalarFile.good())
      throw std::runtime_error("error in opening scalar file " +fileName);
    scalarFile.seekg(0,scalarFile.end);
    size_t numElements = scalarFile.tellg();
    scalarFile.seekg(0,scalarFile.beg);
      
    ScalarField::SP sf = std::make_shared<ScalarField>();
    sf->name = fieldName;
    sf->value.resize(numElements);
    scalarFile.read((char *)sf->value.data(),sizeof(float)*numElements);
#if 1
    std::mutex mutex;
    parallel_for_blocked(0ull,sf->value.size(),1024*1024,[&](size_t begin, size_t end){
        interval<float> blockRange;
        for (size_t i=begin;i<end;i++) {
          float f = sf->value[i];
          blockRange.extend(f);
        }
        std::lock_guard<std::mutex> lock(mutex);
        sf->valueRange.extend(blockRange.lower);
        sf->valueRange.extend(blockRange.upper);
      });
#else
    for (auto f : sf->value)
      sf->valueRange.extend(f);
#endif
    std::cout << "#exa: done loading scalar field '" << fieldName
              << "' from " << fileName << std::endl;
    std::cout << "      (value range " << sf->valueRange << ")" << std::endl;
    return sf;
  }

  ScalarField::SP ScalarField::loadAndComputeMagnitude(const std::string &fieldName,
                                                       const std::string &fnx,
                                                       const std::string &fny,
                                                       const std::string &fnz
                                                       )
  {
    ScalarField::SP in_x = ScalarField::load(fieldName+".x",fnx);
    ScalarField::SP in_y = ScalarField::load(fieldName+".y",fny);
    ScalarField::SP in_z = ScalarField::load(fieldName+".z",fnz);
    assert(in_x->value.size() == in_y->value.size());
    assert(in_x->value.size() == in_z->value.size());
      
    size_t numElements = in_x->value.size();
    ScalarField::SP sf = std::make_shared<ScalarField>();
    sf->name = fieldName;
    sf->value.resize(numElements);
#if 1
    std::mutex mutex;
    parallel_for_blocked(0ull,sf->value.size(),1024*1024,[&](size_t begin, size_t end){
        interval<float> blockRange;
        for (size_t i=begin;i<end;i++) {
          float f = length(vec3f(in_x->value[i],in_y->value[i],in_z->value[i]));
          sf->value[i] = f;
          blockRange.extend(sf->value[i]);
        }
        std::lock_guard<std::mutex> lock(mutex);
        sf->valueRange.extend(blockRange.lower);
        sf->valueRange.extend(blockRange.upper);
      });
#else
    for (int i=0;i<numElements;i++) {
      float f = length(vec3f(in_x->value[i],in_y->value[i],in_z->value[i]));
      sf->value[i] = f;
      sf->valueRange.extend(f);
    }
#endif
    std::cout << "#exa: done computing magnitude of vector field '" << fieldName << std::endl; 
    std::cout << "      (value range " << sf->valueRange << ")" << std::endl;
    return sf;
  }
    
  ScalarField::SP ScalarField::createFromExpression(const std::string &fieldName,
                                                    const std::vector<ScalarField::SP> &fields,
                                                    const std::vector<std::string> &tokens
                                                    )
  {
    auto trim = [](const std::string &in, char c) {
      std::string s = in;
      s.erase(s.begin(), std::find_if(s.begin(), s.end(), [c](unsigned char ch) {
        return ch != c;
      }));
      return s;
    };

    auto rtrim = [](const std::string &in, char c) {
      std::string s = in;
      s.erase(std::find_if(s.rbegin(), s.rend(), [c](unsigned char ch) {
        return ch != c;
      }).base(), s.end());
      return s;
    };

    std::vector<std::string> trimmed(tokens.size());
    for (size_t i=0; i<tokens.size(); ++i) {
        std::string token = tokens[i];
        token = trim(token,'\"');
        token = rtrim(token,'\"');
        token = trim(token,' '); // only ' ' whitespace allowed!
        token = rtrim(token,' '); // only ' ' whitespace allowed!
        trimmed[i] = token;
    }


    size_t numElements = fields[0]->value.size();
    ScalarField::SP sf = std::make_shared<ScalarField>();
    sf->name = fieldName;
    sf->value.resize(numElements);

    std::mutex mutex;
    parallel_for_blocked(0ull,sf->value.size(),1024*1024,[&](size_t begin, size_t end){
        interval<float> blockRange;
        for (size_t i=begin;i<end;i++) {

          float f = 0.f;

          // Postfix eval
          std::stack<float> st;

          for (size_t ti=0; ti<trimmed.size(); ++ti) {
            std::string token = trimmed[ti];//std::cout << token << '\n';
            if (token.empty()) {
              std::cerr << "Empty token\n";
              break;
            }


            if (token[0] == '%') { // Placeholder (%0,%1,%2,...)
              std::string t = trim(token,'%');
              unsigned field = std::stoi(t);

              if (field >= fields.size()) {
                std::cerr << "Invalid placeholder token: " << token << '\n';
                break;
              }

              st.push(fields[field]->value[i]);
            } else if (token == "select") { // ternary token
              float op2 = st.top(); st.pop();
              float op1 = st.top(); st.pop();
              int mask = st.top(); st.pop();

              st.push(mask ? op1 : op2);
            } else if (token == "+" || token == "-" || token == "*" || token == "/" || token == "**"
                    || token == "==" || token == "!=" || token == "<" || token == ">" || token == "<=" || token == ">=") { // binary tokens
              if (st.size() < 2) {
                std::cerr << "Insufficient operands for token: " << token << '\n';
                break;
              }

              float op2 = st.top(); st.pop();
              float op1 = st.top(); st.pop();

              if (token == "+")       st.push(op1+op2);
              else if (token == "-")  st.push(op1-op2);
              else if (token == "*")  st.push(op1*op2);
              else if (token == "/")  st.push(op1/op2);
              else if (token == "**") st.push(powf(op1,op2));
              else if (token == "==")  st.push(op1==op2);
              else if (token == "!=")  st.push(op1!=op2);
              else if (token == "<")  st.push(op1<op2);
              else if (token == ">")  st.push(op1>op2);
              else if (token == "<=")  st.push(op1<=op2);
              else if (token == ">=")  st.push(op1>=op2);
            } else if (token == "log" || token == "abs" || token == "sqrt") {
              if (st.size() < 1) {
                std::cerr << "Insufficient operands for token: " << token << '\n';
                break;
              }

              float op = st.top(); st.pop();

              if (token == "log")       st.push(log(op));
              else if (token == "abs")  st.push(fabsf(op));
              else if (token == "sqrt") st.push(sqrtf(op));
            } else { // constant
              try {
                st.push((float)stod(token));
              } catch (...) {
                std::cerr << "Not a floating point token: " << token << '\n';
              }
            }
          }

          if (st.size() != 1) {
            std::cerr << "Invalid expression\n";
            return;
          }

          f = st.top(); st.pop();
          sf->value[i] = f;
          blockRange.extend(sf->value[i]);
        }
        std::lock_guard<std::mutex> lock(mutex);
        sf->valueRange.extend(blockRange.lower);
        sf->valueRange.extend(blockRange.upper);
      });

    std::cout << "#exa: done computing postfix expression '" << fieldName << '\'' << std::endl; 
    std::cout << "      (value range " << sf->valueRange << ")" << std::endl;
    return sf;
  }

} // ::exa
