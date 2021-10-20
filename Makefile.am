SUBDIRS          = src test utils
AUTOMAKE_OPTIONS = foreign
ACLOCAL_AMFLAGS  = -I m4
EXTRA_DIST       = CHANGES LICENSE dist_version

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
