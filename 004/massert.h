//http://www.devarticles.com/c/a/Cplusplus/C-plus-plus-Preprocessor-Always-Assert-Your-Code-Is-Right/2/
#ifndef MASSERT_H
#define MASSERT_H

//#define DEBUG

#include <stdlib.h>
#include <iostream>

#ifndef DEBUG
# define MASSERT( unused ) do {} while ( false )    
#else
  #define MASSERT(isOK,msg) \
    if ( !(isOK) ) \
    { \
      std::cout << "In file " << __FILE__ << ":"; \
      std::cout << __LINE__ << ":"; \
      std::cout << " Error !! Assert { " << #isOK << " } failed\n"; \
      std::cout << " Message : " << msg << std::endl; \
      abort(); \
    }
#endif

/*
#ifndef DEBUG
# define MASSERT_MSG( unused ) do {} while ( false )    
#else
  #define MASSERT_MSG(msg) \
    { \
      std::cout << "MSG: " << msg << std::endl; \
    }
#endif
//*/

#endif