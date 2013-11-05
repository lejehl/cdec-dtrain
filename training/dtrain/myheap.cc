/*
 * myheap.cc
 *
 *  Created on: Apr 19, 2013
 *      Author: laura
 */

#include <algorithm>

#include "myheap.h"



using namespace std;


MyHeap::MyHeap( unsigned size ): heap_(  )
{
	is_full_ = false;
	size_ = size;
//	cout << "Made new heap" << ", size: " << heap_.size() << endl;
	make_heap(heap_.begin(), heap_.end(), sort_by_second() );

}

/*
* 1. assume that heap is sorted
* 2a. If heap is not yet full: add element
* 2b. else: if value of first element in heap > pair Value: stop,
* else add element to heap
*  3.Check if heap is full.
* 4. Sort heap
*/
void MyHeap::addPair( pair<string, double> & new_pair ){
//	cout << "Trying to add pair: " << "<" << new_pair.first << "," << new_pair.second << ">" << endl;
	if ( is_full_ ){
//		sort_heap( heap_.begin(), heap_.end() );
//		if ( heap_.top()->second < new_pair.second ) {
//			heap_.pop();
//			heap_.push(new_pair );
//		}
		if ( heap_.front().second  < new_pair.second  ) {
//			cout << "initial max heap   : " << heap_.front().second << '\n';
			pop_heap(heap_.begin(), heap_.end(), sort_by_second() ); heap_.pop_back();
//			cout << "max heap after pop_back   : " << heap_.front().second << '\n';
			heap_.push_back( new_pair ); push_heap( heap_.begin(), heap_.end(), sort_by_second() );
//			cout << "max heap after push_back   : " << heap_.front().second << '\n';
		}
	} else {
		heap_.push_back(new_pair); push_heap( heap_.begin(), heap_.end(), sort_by_second() );
//		cout << "max heap after push_back   : " << heap_.front().second << '\n';
	}
//		if ( heap_.at(0).second > new_pair.second ){
//			cout << "smallest element is greater then new pair. Not adding pair." << endl;

//		} else {
//			cout << "Pair is added" << endl;
//			heap_.at(0)= new_pair;
//		}
//	} else {
//		cout << "Heap not full yet: Adding pair" << endl;
//		vector<pair<string, double> >:: iterator it = heap_.begin();
//		heap_.insert( it, new_pair );
//	}
//	sortByVal();
	checkIfFull();
//	printHeap();


}

void MyHeap::printHeap(){
	cout << "Heap: ";
	for ( unsigned i = 0; i < heap_.size(); i++ ){
		cout << "<" << heap_.at(i).first << ", " << heap_.at(i).second << ">, ";
	}
	cout << endl;
}

//void MyHeap::sortByVal(){
//	sort( heap_.begin(), heap_.end(), sort_by_second() );
//
//}

void MyHeap::checkIfFull()
{
	if ( heap_.size() < size_ ){
		is_full_ = false;
	}
	else is_full_ = true;

}


void MyHeap::reverseHeap( std::vector< std::pair<string, double> >& rev_heap )
{
//	reverse( heap_.begin(), heap_.end());
//	cout << "before sorting: " << endl ;
//	printHeap();
	sort_heap(heap_.begin(), heap_.end(), sort_by_second() );
	cout << "after sorting: " << endl ;
	printHeap();
	for ( unsigned i=0; i< heap_.size(); i++ ) {
		rev_heap.push_back( heap_[i]);
	}
}
