#include <queue>
#include <calibn/calibgraph.hpp>
#include <calibn/helpers.hpp>

using namespace std;

// a little helper comparator for ultilising STL priority queue to solve spanning tree problem
struct CmpCalibPairs {bool operator()(CalibPair* const & p1, CalibPair* const & p2) {return p1->GetSize() < p2->GetSize();}};

bool CalibGraph::Create(const CalibCameras& cams, size_t startingCam)
{
    //m_report = report;
	m_cams = cams;
	m_numCams = cams.size();
	m_startingCamIdx = startingCam;

	// contruct all possible calibration pairs including invalid ones that share no sufficient calibration images
	for (CalibCameras::iterator cam0 = m_cams.begin(); cam0 != m_cams.end(); cam0++)
	{
		for(CalibCameras::iterator cam1 = boost::next(cam0); cam1 != m_cams.end(); cam1++)
		{
			m_pairs.push_back(CalibPair(&(*cam0), &(*cam1)));
		}
	}

	std::priority_queue<CalibPair*, CalibPairPtrs, CmpCalibPairs> Q; // priority queue storing the explored edges
	size_t NiL = (size_t) -1; // denoting the null vertex

	CalibIndexList pre(m_numCams, NiL); // precdecessors for backtracing
	
	// Prim's algorithm
	// 1. Initialisation
	CalibPairPtrs initPairs = GetPairsByCam(m_startingCamIdx);
	for (CalibPairPtrs::iterator itr = initPairs.begin(); itr != initPairs.end(); itr++)
	{
		Q.push((*itr));
	}

	// 2. Building MST
	while (!Q.empty())
	{
		CalibPair* pair = Q.top();
		Q.pop();

		size_t p = pair->GetCamera1()->GetIndex();
		size_t q = pair->GetCamera2()->GetIndex();

		bool p_visited = (p == m_startingCamIdx || pre[p] != NiL);
		bool q_visited = (q == m_startingCamIdx || pre[q] != NiL);

		assert(p_visited || q_visited);

		if (p_visited && q_visited) continue; // shall we ignore the edge if adding it leads to loop
		if (q_visited) swap(p,q); // make sure we traverse in the right direction of p->q

		pre[q] = p;
		pair->SetEnabled(true);

		CalibPairPtrs pairs = GetPairsByCam(q);
		for (CalibPairPtrs::iterator itr = pairs.begin(); itr != pairs.end(); itr++)
		{
			Q.push((*itr));
		}
	}

	// 3. Backtracing to bulid traversal sequence starting at cam1 to cam2, cam3..
	m_seqs = CalibListOfIndexList(m_numCams);
	for (size_t q = 0; q < m_numCams; q++)
	{
		size_t p = pre[q];

		if (p == NiL && q != m_startingCamIdx) // this camera is non-reacheable from cam0
		{
			m_seqs.clear();
			return false;
		}

		while (p != NiL)
		{
			m_seqs[q].push_back(p);
			p = pre[p];
		}
	}
	
	return true;
}

bool CalibGraph::Initialise(bool pairOptim)
{
	if (!IsOkay())
	{
		return false;
	}

	// to enable or disable pairwise optimisation
	Calibratable::OptimTermCriteria = pairOptim ?
	    m_termCriteria : cv::TermCriteria(cv::TermCriteria::COUNT, 0, 0);

	m_params.Extrinsics = CalibExtrinsicsList(m_seqs.size());
	m_params.Intrinsics = CalibIntrinsicsList(m_numCams);

	double rpeSum = 0;

	E_INFO << "starting pair-wise initialisation..";

	for (size_t cam = 0; cam < m_seqs.size(); cam++)
	{
		size_t p = cam;
		for (CalibIndexList::iterator itr = m_seqs[cam].begin(); itr != m_seqs[cam].end(); itr++)
		{
			size_t q = *itr;
			CalibPair& e = *GetPair(p,q);
			bool inverseExtrinsics = (e.GetCamera1()->GetIndex() != p);  // u -> v

			if (!e.IsCalibrated())
			{
				E_INFO << "calibrating pair (" << p << "," << q << ") from " << e.GetSize() << " images..";

				double rpe = e.Calibrate();
				if (rpe < 0)
				{
					printf("..FAILED\n");
					cerr << "Pair-wise initialisation of (" << p << "," << q << ") failed" << endl;
					return false;
				}

				printf("..DONE, RPE=%.02f\n", rpe);
				rpeSum += rpe;
			}

			m_params.Extrinsics[cam] += e.GetExtrinsics(inverseExtrinsics);
			p = q; // proceed to next path
		}

		assert(p == m_startingCamIdx);
		m_params.Extrinsics[cam].Inverse(); // from the starting cam to cam #n
		m_params.Intrinsics[cam] = m_cams[cam].GetIntrinsics();
	}

	m_params.ImagePoses = DeriveImagePoses(m_pairs, m_params.Extrinsics);
	//m_report.WriteVisibilityGraph(m_pairs, m_cams, m_seqs);

	return true;
}

CalibExtrinsicsList CalibGraph::DeriveImagePoses(CalibPairs& pairs, CalibExtrinsicsList& extrinsics)
{
	size_t numImages = pairs.front().GetCamera1()->GetSize();
	CalibExtrinsicsList imagePoses(numImages);
	vector<double> bestRpe = vector<double>(numImages);

	for (CalibPairs::iterator pair = pairs.begin(); pair != pairs.end(); pair++)
	{
		if (!pair->IsCalibrated())
		{
			continue;
		}

		double pairRpe = pair->GetRpe();
		vector<CalibCam*> pairCams = vector<CalibCam*>(2);
		pairCams[0] = pair->GetCamera1();
		pairCams[1] = pair->GetCamera2();

		for (vector<CalibCam*>::iterator itrCamPtr = pairCams.begin(); itrCamPtr != pairCams.end(); itrCamPtr++)
		{
			size_t cam = (*itrCamPtr)->GetIndex();
			CalibIndexList imageIndices = (*itrCamPtr)->GetImageIndices();

			for (CalibIndexList::iterator itrImgIdx = imageIndices.begin(); itrImgIdx != imageIndices.end(); itrImgIdx++)
			{
				size_t img = *itrImgIdx;
				bool better = (bestRpe[img] == 0 || pair->GetRpe() < bestRpe[img]);

				if (!better) continue;

				CalibExtrinsics pose = (*itrCamPtr)->GetImagePose(img);
				imagePoses[img] = pose.Concatenate(extrinsics[cam].GetInverse());
				bestRpe[img] = pair->GetRpe();
			}
		}
	}

	return imagePoses;
}

cv::Mat CalibGraph::Evaluate(const cv::Mat& x, const Mask& mask) const
{
	CalibBundleParams params(m_numCams, m_startingCamIdx);
	vector<double> y;

	bool success = params.FromVector(x);
	assert(success);

	if (!mask.empty())
	{
		y.reserve(mask.size());
	}

	for (CalibCameras::const_iterator camItr = m_cams.begin(); camItr != m_cams.end(); camItr++)
	{
		CalibIndexList imageIndices = camItr->GetImageIndices();
		size_t cam = camItr->GetIndex();

		for (CalibIndexList::iterator imgIdxItr = imageIndices.begin(); imgIdxItr != imageIndices.end(); imgIdxItr++)
		{
			size_t img = *imgIdxItr;
			CalibData data = camItr->GetData(img);

			assert (data.IsOkay());

			ImagePointList ip = data.GetImagePoints();
			bool eval = true;

			if (!mask.empty()) // look ahead, check if we should skip this data chunk
			{
				for (size_t i = 0; i < ip.size(); i++)
				{
					if (eval = mask[y.size() + i]) break;
				}
			}

			if (eval)
			{
				ImagePointList rp = params.Project(data.GetObjectPoints(), cam, img);
				for (size_t i = 0; i < ip.size(); i++)
				{
					y.push_back(rp[i].x - ip[i].x);
					y.push_back(rp[i].y - ip[i].y);
				}
			}
			else
			{
				for (size_t i = 0; i < ip.size(); i++)
				{
					y.push_back(0);
					y.push_back(0);
				}
			}
		}
	}

	return cv::Mat(y).clone();
}

void CalibGraph::BuildRpeData(const cv::Mat& y, std::vector<std::vector<ImagePointList> >& rpeData) const
{
	rpeData = std::vector<std::vector<ImagePointList> >(m_cams.size());
	size_t k = 0;

	for (CalibCameras::const_iterator camItr = m_cams.begin(); camItr != m_cams.end(); camItr++)
	{
		CalibIndexList images = camItr->GetImageIndices();
		size_t cam = camItr->GetIndex();

		rpeData[cam] = vector<ImagePointList>(camItr->GetSize());

		for (CalibIndexList::iterator imgIdxItr = images.begin(); imgIdxItr != images.end(); imgIdxItr++)
		{
			size_t img = *imgIdxItr;
			size_t numPoints = camItr->GetData(img).GetImagePoints().size();

			assert(k + numPoints * 2 <= y.rows); // boundary check

			rpeData[cam][img] = ImagePointList(numPoints);
			for (size_t i = 0; i < numPoints; i++)
			{
				rpeData[cam][img][i].x = (float) y.at<double>(k++);
				rpeData[cam][img][i].y = (float) y.at<double>(k++);
			}
		}
	}

	assert(k == y.rows);
}

void CalibGraph::Finalise(const CalibOptimState& state)
{
	// restore the optimised parameteres
	m_params.FromVector(state.x);

	// approximate standard errors of the estimated parameters
	CalibBundleParams sigma(m_numCams, m_startingCamIdx);
	sigma.FromVector(state.sigma);

	// build data for RPE plotting
	vector<vector<ImagePointList> > rpeData;
	BuildRpeData(state.y, rpeData);

    /*
	m_report.WriteParams(m_params, sigma);
	m_report.WriteOptimisationPlot(state);
	m_report.WriteHessianMatrix(state.H);
	m_report.WriteCoverageImages(m_cams);
	m_report.WriteFusionImage(m_cams, m_params);
	m_report.WriteRpePlots(rpeData);
	m_report.WriteRpeImages(m_cams, m_params);
	*/
}

size_t CalibGraph::GetPairIndex(size_t i, size_t j, size_t cams)
{
	assert(i != j);
	size_t p = (i < j) ? i : j;
	size_t q = (i < j) ? j : i;
	size_t n = cams - 1;

	return (2*n - p + 1) * p / 2 + (q-p-1);
}

CalibPairPtrs CalibGraph::GetPairsByCam(size_t cam)
{
	CalibPairPtrs pairPtrs;

	for (size_t p = 0; p < m_numCams; p++)
	{
		if (p == cam) continue;
		CalibPair* pair = GetPair(cam, p);
		if (pair->IsOkay()) pairPtrs.push_back(pair);
	}

	return pairPtrs;
}
