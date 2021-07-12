TEST_LOG_DRIVER = env AM_TAP_AWK='$(AWK)' $(SHELL) \
                  $(top_srcdir)/test/tap-driver.sh
if CUDA_ENABLED
TESTS = cyl3d.gpu.test \
	cyl3d.mflow.gpu.test \
	wedge.gpu.test
else
TESTS = cyl3d.test \
	cyl3d.mflow.test \
	wedge.test
endif
