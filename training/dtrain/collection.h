/*
 * collection.h
 *
 *  Created on: Apr 4, 2013
 *      Author: laura
 */

#ifndef COLLECTION_H_
#define COLLECTION_H_

#include <vector>
#include <map>
#include <sstream>

#include <boost/algorithm/string.hpp>

#include "filelib.h"
#include "wordid.h"
#include "document.h"

using namespace std;

struct Collection
{
public:
  Collection( const string& filename );

  string infile_;
  unsigned num_docs_;//collection size
  unsigned getNumDocs();
  void splitOnTabs( string&, vector<string>& );
  void tokenize( string&, vector<string>& );
//	void printVector( vector<string> & );
};

struct DocumentCollection : public Collection
{
public:
  DocumentCollection( string& filename );
  map<string, Document > collection_;
  map<WordID, unsigned> dftable_;
  double avg_len_;
  void loadDocs( );
  void generateDfTable(  );
  void averageDocLength();
  double getAvgLen();
};

struct QueryCollection : public Collection
{

public:
  QueryCollection( string& filename, string& relevance_file, string& sw_file );
  map<string, Query > collection_;
  map<unsigned,string> sentence_qid_map_;
  string relfile_;
  set<WordID> stopwords_;

  void loadQueries( );
  void loadStopwords( string& );
//	void printQueryCollection();
  void setSentence( unsigned, vector<WordID> & );
//	void setSentence( unsigned, string & );
  map<string, Query >::iterator getQuery( unsigned /*sent_id*/ );
};


#endif /* COLLECTION_H_ */
