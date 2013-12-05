/*
 * ff_np.cc
 *
 *  Created on: 04.12.2013
 *      Author: laura
 */

#include "ff_np.h"

#include "filelib.h"
#include "sentence_metadata.h"
#include "pugixml.hpp"


using namespace std;


NPMatchFeatures::NPMatchFeatures(const std::string& param) {
	fid_(FD::Convert("NPMatch")); // feature id (dense feature)
}

/*
 * parse chunked string
 */
void NPMatchFeatures::PrepareForInput(const SentenceMetadata& smeta ) {
	np_spans_.clear();
	std::ifstream is(smeta.GetSGMLValue("src_chunks"));
	// TODO: these should be given as parameters somewhere
	std::string jp_nnp= "名詞,固有名詞"; // proper noun
	std::string jp_nn="名詞,一般"; // common noun

//	ReadFile f = ReadFile(smeta.GetSGMLValue("src_chunks"));
//	string chunks;
//	f.ReadAll( chunks );
	pugi::xml_document doc;
	pugi::xml_parse_result res = doc.load( is );
	pugi::xml_node s=doc.child("Sentence");
	short int pos = 0;
	// for each chunk
	for(pugi::xml_node c = s.first_child; c; c=c.next_sibling()) {
		short int len_chunk=std::distance( c.children().begin(), c.children().end() ) ;
		 // find head token
		pugi::xml_node h = c.find_child_by_attribute( "id", c.attribute("head").value() );
		// get feature string
		std::string feats= string(h.attribute("feature").value());
		// if head is a noun, store as NP
		if ( feats.compare(0, jp_nn.length() , jp_nn ) ||
				feats.compare(0, jp_nnp.length(), jp_nnp ) ) {
				 np_spans_.insert(np_spans_.end(),
						 std::make_pair( pos, len_chunk ));
		}
		pos += len_chunk;
	}
	// parse into np_spans!

}


void NPMatchFeatures::TraversalFeaturesImpl(const SentenceMetadata& smeta,
        const Hypergraph::Edge& edge,
        const vector<const void*>& ant_contexts,
        SparseVector<double>* features,
        SparseVector<double>* estimated_features,
        void* context) const {
	std::pair<short int, short int> span= std::make_pair(edge.i_, edge.j_) ;
	if ( np_spans_.find(span) != np_spans_.end() )
		features->add_value(fid_,1); // augment count if rule spans an NP
}

