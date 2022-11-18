SUBDIRS          = src test utils
AUTOMAKE_OPTIONS = foreign
ACLOCAL_AMFLAGS  = -I m4
EXTRA_DIST       = CHANGES LICENSE dist_version m4/snarf_mfem.py m4/wrap_lines.py src/tps.py

# --------------------------------------------------
# Revision control support for external distribution
# --------------------------------------------------

if GIT_CHECKOUT
dist_version: FORCE
	@GIT_REVISION@ > $(top_srcdir)/dist_version

FORCE:
endif

dist-hook:
	rm -rf `find $(distdir)/ -name .gitattributes`

# -------------------------------------------
# Optional support for code coverage analysis
# -------------------------------------------
if CODE_COVERAGE_ENABLED
gcovr_dir=$(top_builddir)/coverage/html

gcovr-report:
	@mkdir -p $(gcovr_dir)
	gcovr -r $(top_srcdir) --gcov-filter $(top_builddir) --exclude-unreachable-branches --exclude-throw-branches \
	    --html-details $(gcovr_dir)/gcovr-report.html
gcovr-reset:
	@rm -rf $(gcovr_dir)
	@rm -f $(find . -name "*.gcda")

gcovr-clean: gcovr-reset
	@rm -f $(find . -name "*.gcno")

coverage: gcovr-reset check gcovr-report

endif
