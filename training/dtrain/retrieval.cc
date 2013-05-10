/*
 * retrieval.cc
 *
 *  Created on: May 10, 2013
 *      Author: laura
 */

#include <iostream>

#include "retrieval.h"

using namespace std;

double RetrievalEval::averagePrecision( vector<unsigned>& gold, vector<unsigned>& retrieved ){
	unsigned counter = 0;
	double sum = 0.0;

	// precision at i
	cout << "relevance (gold)\tP@rank" << endl;
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



