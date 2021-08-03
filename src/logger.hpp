#ifndef LOGGER
#define LOGGER

#include<grvy.h>

#ifdef DEBUG
#undef DEBUG
#endif

typedef enum {INFO  = GRVY_INFO,
	      DEBUG = GRVY_DEBUG,
	      WARN  = GRVY_WARN
} loglevels;


#endif	// LOGGER
