#ifndef SYSTEM_H
#define SYSTEM_H

#include <stdlib.h>
#include <cstdio>
#include <iostream>

//#include <cstring>
#include <cstdarg>
//#include <ctime>
//#include <cstdlib>
//#include <sys/stat.h>
//#include <cerrno>
//#include <sstream>

//#include <fstream>
//#include <string>

using namespace std;


// **************************************************************
/*!  @brief  Identifiers for various output messages.
 */
typedef enum MessageType {
    Msg = 0, MsgDbg, MsgLog, MsgVerb, Warn, UsrErr, Err, PrgErr
} MessageType;


#define xprintf(...) _xprintf(__FILE__, __func__, __LINE__, __VA_ARGS__)

int     _xprintf(const char * const xprintf_file, const char * const xprintf_func, const int xprintf_line, MessageType type, const char * const fmt, ... );

string create_subdirectory(string parent_dir, string new_dir);


//http://www.devarticles.com/c/a/Cplusplus/C-plus-plus-Preprocessor-Always-Assert-Your-Code-Is-Right/2/
//#ifndef MASSERT_H
//#define MASSERT_H

//#define DEBUG

#include <stdlib.h>
#include <iostream>

#ifndef DEBUG
# define MASSERT( unused1, unused2 ) do {} while ( false ) 
# define DBGMSG(...)
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

#define DBGMSG(...) do { xprintf(MsgDbg,__VA_ARGS__); fflush(NULL); } while (0)

#endif

#endif 
