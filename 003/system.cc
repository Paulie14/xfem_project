

#include "system.hh"

//create directory
#include <dirent.h>
#include <sys/stat.h>
#include <sstream>


/// @brief INTERNAL DEFINITIONS FOR XPRINTF
/// @{

struct MsgFmt {
  int  num;           ///< format number
  bool log;       ///< log the message - YES/NO
  bool mpi;           ///< treat as global message (invoke MPI_Barrier() when printing)
  int screen;         ///< print to stdout,stderr,NULL
    bool stop;          ///< terminate the program
  const char * head;  ///< message formating string
};

#define SCR_NONE  0
#define SCR_STDOUT  1
#define SCR_STDERR  2

/// configuration table for individual message types defined in system.h
/// Msg type    Log    mpi      screen      Stop    message header
#define NUM_OF_FMTS   8
static struct MsgFmt msg_fmt[] = {
  {Msg,     true,  false,   SCR_STDOUT, false,  NULL},
  {MsgDbg,    true,  false,   SCR_STDOUT, false,  "DBG (%s, %s(), %d):"},
  {MsgLog,  true,  false,   SCR_NONE, false,  NULL},
  {MsgVerb, false, false,   SCR_STDOUT, false,  NULL},
  {Warn,    true,  false,   SCR_STDERR, false,  "Warning (%s, %s(), %d):\n"},
  {UsrErr,  true,  false,   SCR_STDERR, true, "User Error (%s, %s(), %d):\n"},
  {Err,   true,  false,   SCR_STDERR, true, "Error (%s, %s(), %d):\n"},
  {PrgErr,  true,  false,   SCR_STDERR, true, "Internal Error (%s, %s(), %d):\n"}
};


/*!
 * @brief Multi-purpose printing routine: messages, warnings, errors
 * @param[in] xprintf_file   current file
 * @param[in] xprintf_func   current function
 * @param[in] xprintf_line   current line number
 * @param[in] type           message type
 * @param[in] fmt            message format
 * @return      Same as printf, what internal printing routine returns.
 */
int _xprintf(const char * const xprintf_file, const char * const xprintf_func, const int xprintf_line, MessageType type, const char * const fmt, ... )
{
  struct MsgFmt mf;
  int rc;
  FILE *screen=NULL;


  if ((type < 0) || (type >= NUM_OF_FMTS))
    type = Msg;
  mf = msg_fmt[type];

  // determine output stream
  switch (mf.screen) {
    case SCR_STDOUT : screen=stdout; break;
    case SCR_STDERR : screen=stderr; break;
    case SCR_NONE : screen=NULL; break;
    default: screen=NULL;
  }


    //if not PRINTALL, allow console output only for MPI master, no need to print mpi_id
  //if ( (screen)  )
  //      screen = NULL;

  // print head
  if (mf.head) {
    if (screen) fprintf(screen,mf.head,xprintf_file,xprintf_func,xprintf_line);
  }
  // print message
  {
    va_list argptr;

    if (screen)
    {
      va_start( argptr, fmt );
      rc=vfprintf(screen,fmt,argptr);
      va_end( argptr );
        // flush every message (maybe there is a problem in cygwin without that)
      fflush(screen);
    }
  }
  return rc;
}


std::string create_subdirectory(std::string parent_dir, std::string new_dir)
{
    std::stringstream dir_name;
    dir_name << parent_dir << "/" << new_dir << "/";
    DIR *dir;
    dir = opendir(dir_name.str().c_str());
    if(dir == NULL) {
        int ret = mkdir(dir_name.str().c_str(), 0777);

        if(ret != 0) {
            xprintf(Err, "Couldn't create directory: %s\n", dir_name.str().c_str());
        }
    } else {
        closedir(dir);
    }
    return dir_name.str();
}