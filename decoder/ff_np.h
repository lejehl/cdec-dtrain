/*
 * ff_np.h
 *
 *  Created on: 04.12.2013
 *      Author: laura
 */

#ifndef FF_NP_H_
#define FF_NP_H_

#include "ff.h"
#include "hg.h"
#include "array2d.h"
#include "wordid.h"

struct NPMatchFeaturesImpl;

class NPMatchFeatures : public FeatureFunction {
public:
	NPMatchFeatures(const std::string& param);
	~NPMatchFeatures();
protected:
	virtual void TraversalFeaturesImpl(const SentenceMetadata& smeta,
									   const HG::Edge& edge,
									   const std::vector<const void*>& ant_contexts,
									   SparseVector<double>* features,
									   SparseVector<double>* estimated_features,
									   void* context) const;
	virtual void PrepareForInput(const SentenceMetadata& smeta );
private:
	std::set< std::pair<short int, short int> > np_spans_; // spans of NPs in chunking of input
	const int fid_;
};

#endif /* FF_NP_H_ */
