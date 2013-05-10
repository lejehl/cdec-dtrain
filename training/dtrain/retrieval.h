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
	double averagePrecision( vector<unsigned>& gold, vector<unsigned>& retrieved  );

};


#endif /* RETRIEVAL_H_ */
