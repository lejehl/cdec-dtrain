/*
 * myheap.h
 *
 *  Created on: Apr 19, 2013
 *      Author: laura
 */

#ifndef MYHEAP_H_
#define MYHEAP_H_

#include <boost/algorithm/string.hpp>
#include <iostream>
#include <vector>
#include <map>

using namespace std;

struct sort_by_second {
    bool operator()(const pair< string, double > &left, const pair< string, double > &right) {
        return left.second < right.second;
    }
};

struct MyHeap
{
public:
	MyHeap( unsigned );
	vector< pair<string, double> > heap_;
	unsigned size_;
	// add key value pair
	void addPair( pair<string, double>& );
		// 1. assume that heap is sorted
		// 2a. If heap is not yet full: add element
		// 2b. else: if value of first element in heap > pair Value: stop,
		//else add element to heap
		// 3.Check if heap is full.
		// 4. Sort heap
	void printHeap();
	void reverseHeap();
private:
	bool is_full_;
	// sort vector by value
	void sortByVal( );
	void checkIfFull();



};



#endif /* MYHEAP_H_ */
