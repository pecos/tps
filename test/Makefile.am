TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' $(SHELL) $(top_srcdir)/test/tap-driver.sh

EXTRA_DIST      = tap-driver.sh soln_differ inputs meshes ref_solns/*.h5 \
		  vpath.sh die.sh cyl3d.gpu.test cyl3d.mflow.gpu.test wedge.gpu.test \
	          averages.gpu.test cyl3d.test cyl3d.mflow.test cyl3d.dtconst.test \
		  die.test averages.test wedge.test cyl3d.interp.test valgrind.test

TESTS = vpath.sh

if GPU_ENABLED
TESTS += cyl3d.gpu.test \
	 cyl3d.mflow.gpu.test \
	 wedge.gpu.test \
	 averages.gpu.test
else
TESTS += cyl3d.test \
	 cyl3d.mflow.test \
	 cyl3d.dtconst.test \
	 die.test \
	 averages.test \
	 wedge.test \
	 qms.rings.test
endif

if GSLIB_ENABLED
TESTS += cyl3d.interp.test
endif

if VALGRIND_ENABLED
check_PROGRAMS = uninit
uninit_SOURCES = badloop.cpp

TESTS  += valgrind.test
endif
