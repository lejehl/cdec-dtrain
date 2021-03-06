/*
 * test-retrieval.cc
 *
 *  Created on: May 15, 2013
 *      Author: laura
 */

#include <iostream>
#include <string.h>
#include <vector>
#include <map>
#include <set>

#include "retrieval.h"
#include "collection.h"
#include "viterbiget.h"
#include "decoder.h"
#include "ff_register.h"
#include "myheap.h"
#include "document.h"


//#include "wordid.h"

using namespace std;

int main( int argc, char** argv){
register_feature_functions();
SetSilent(true);
ReadFile ini_rf( "cdec.conf" );
Decoder decoder(ini_rf.stream());

ReadFile input("queries.seg");
string docfile = "/workspace/dmap/test-mapscore/tune_small_set/docs.preproc.en";
string qfile = "queries.split.jp";
string relfile = "dev.qrel";

DocumentCollection docs( docfile );
QueryCollection queries( qfile, relfile);

// decode
cerr << "setting viterbi translations" << endl;
dtrain::ViterbiGetter* v = new dtrain::ViterbiGetter();
string in;
unsigned it = 0;
while( getline(*input, in) ) {
	decoder.Decode( in, v);
	queries.setSentence( it, v->transl_ );
	it ++;
}
cout << "run retrieval:" << endl;
Retrieval R;
MyHeap results(10);
string id = "JP-2006000633-A";
cout << queries.collection_.size() << endl;
queries.collection_.at( id ).setTerms();
time_t start, end;
time(&start);
R.runRetrieval( queries.collection_.at( id ).terms_, docs, results   );
time(&end);
float time_diff = difftime(end, start);

R.evaluateRetrieval( queries.collection_.at( id ).relevant_docs_, results );
cout << "time elapsed: " << time_diff << " second" << endl;
}

