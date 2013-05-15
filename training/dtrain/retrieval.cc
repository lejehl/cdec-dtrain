/*
 * retrieval.cc
 *
 *  Created on: May 10, 2013
 *      Author: laura
 */

#include <iostream>
#include <stdlib.h>

#include "retrieval.h"
#include "collection.h"
#include "document.h"
#include "wordid.h"

using namespace std;

double RetrievalEval::averagePrecision( vector<unsigned>& gold, vector<unsigned>& retrieved ){
	unsigned counter = 0;
	double sum = 0.0;

	// precision at i
	cout << "relevance (gold)\tsum(precision@rank)" << endl;
	for ( unsigned i = 0; i < gold.size(); i++ ){
		if ( retrieved.at(i) >= gold.at(i) ){
			counter++;
			sum += (double) counter / (double)( i+1 );
		}
		cout << retrieved.at(i) << " (" << gold.at(i) << ") " ;
	}
	cout << "\t" << sum;
	double avPrec;

	// normalize by number of RELEVANT docs
	avPrec = sum/gold.size();
//	cout << "\t" << avPrec;
	cout << endl;
	return avPrec;

}

double RetrievalEval::avPrecAtN( vector<unsigned>& gold, vector<unsigned>& retrieved, unsigned n ){
	double avPrec = 0.0;
	if  (  n > retrieved.size()  )  {
		cerr << "n must be less than or equal to num_retrieved!" << endl;
		exit(1);
	} else if ( n == 0 ) {
		n = gold.size();
	} else {
		if (n < gold.size() ) {
				cout << "retrieved has " << retrieved.size() << " elements." << endl;
				cout << "gold has " << gold.size() << " elements." << endl;
				gold.erase( gold.begin()+n, gold.end() );
		}
		if ( n > gold.size() ) {
			for ( unsigned i = gold.size(); i < n; i++ ){
				gold.push_back( 0 );
			}
		}
	}
	avPrec = averagePrecision( gold, retrieved );

	return avPrec;
}

//TODO
//void Retrieval::Retrieval( string query_file, string doc_file, string relevance_file, unsigned int heap_size )
//: docs_( doc_file ), queries_( query_file, relevance_file ), eval_(){
//	heap_size_ = heap_size;
//}
//
//void Retrieval::retrieval( DocumentCollection& docs, set<WordID>& query, MyHeap& results ){
//	for ( map<string, Document>::iterator docIter = docs.collection_.begin();
//			docIter != docs.collection_.end(); ++docIter ){
//		double score = 0.0;
//		// TODO: query should have a term iterator?
//		for ( set<WordID>::iterator it = query.begin(); it != query.end(); ++it ){
//			score += docIter->second.getScoreForQueryTerm( *it );
//		}
//		// add to heap if score is greate than 0
//		if (score != 0.0){
//		pair<string, double> p = make_pair( docIter->first, score );
//		results.addPair( p );
//
//		}
//
//	}
//}
