#ifndef LOGGER
#define LOGGER

#include<grvy.h>

typedef enum {ginfo  = GRVY_INFO,
	      gdebug = GRVY_DEBUG,
	      gwarn  = GRVY_WARN,
              gerror = GRVY_ERROR
} loglevels;


#endif	// LOGGER
