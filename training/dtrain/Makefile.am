bin_PROGRAMS = dtrain 

dtrain_SOURCES = dtrain.cc score.cc dtrain.h kbestget.h ksampler.h pairsampling.h score.h document.h document.cc collection.h collection.cc myheap.h myheap.cc viterbiget.h
dtrain_LDADD   = ../../decoder/libcdec.a ../../klm/search/libksearch.a ../../mteval/libmteval.a ../../utils/libutils.a ../../klm/lm/libklm.a ../../klm/util/libklm_util.a ../../klm/util/double-conversion/libklm_util_double.a

#test_collection_SOURCES = collection.cc document.cc test-collection.cc collection.h  document.h
#test_collection_LDADD   = ../../decoder/libcdec.a ../../klm/search/libksearch.a ../../mteval/libmteval.a ../../utils/libutils.a ../../klm/lm/libklm.a ../../klm/util/libklm_util.a ../../klm/util/double-conversion/libklm_util_double.a

#test_document_SOURCES = document.cc test-document.cc document.h
#test_document_LDADD   = ../../decoder/libcdec.a ../../klm/search/libksearch.a ../../mteval/libmteval.a ../../utils/libutils.a ../../klm/lm/libklm.a ../../klm/util/libklm_util.a ../../klm/util/double-conversion/libklm_util_double.a

#test_heap_SOURCES = myheap.cc test-heap.cc myheap.h
#test_heap_LDADD   = ../../decoder/libcdec.a ../../klm/search/libksearch.a ../../mteval/libmteval.a ../../utils/libutils.a ../../klm/lm/libklm.a ../../klm/util/libklm_util.a ../../klm/util/double-conversion/libklm_util_double.a


AM_CPPFLAGS = -W -Wall -Wno-sign-compare -I$(top_srcdir)/utils -I$(top_srcdir)/decoder -I$(top_srcdir)/mteval

