/*
 * retrieval.h
 *
 *  Created on: May 10, 2013
 *      Author: laura
 */

#ifndef RETRIEVAL_H_
#define RETRIEVAL_H_

#include <vector>


using namespace std;

struct RetrievalEval {
	RetrievalEval() {};

	double avPrecAtN( vector<unsigned>& gold, vector<unsigned>& retrieved, unsigned n = 0 );
private:
	double averagePrecision( vector<unsigned>& gold, vector<unsigned>& retrieved  );

};

//TODO
//struct Retrieval {
//	Retrieval( string query_file, string doc_file, string relevance_file, unsigned heap_size = 10 );
// 	DocumentCollection docs_;
// 	QueryCollection queries_;
// 	RetrievalEval eval_;
// 	unsigned heap_size_;
// 	void retrieval( set<WordID>& query, MyHeap& results );
//
//
//};
#endif /* RETRIEVAL_H_ */
