#include <iomanip>
#include "calibgraphbuilder.hpp"
#include <boost/algorithm/string.hpp>

bool CalibGraphBuilder::TargetDef::FromString(const String& def)
{
    Strings toks = explode(def, 'x');

    if (toks.size() != 3 && toks.size() != 4)
    {
        return false;
    }

    bool success = true;

    success &= sscanf(toks[0].c_str(), "%d", &patternSize.height)   == 1;
    success &= sscanf(toks[1].c_str(), "%d", &patternSize.width)    == 1;
    success &= sscanf(toks[2].c_str(), "%f", &patternMetric.height) == 1;

    if (toks.size() == 3)
    {
        patternMetric.width = patternMetric.height;
    }
    else
    {
        success &= sscanf(toks[2].c_str(), "%f", &patternMetric.width) == 1;
    }

    return success && IsOkay();
}

String CalibGraphBuilder::TargetDef::ToString() const
{
    std::stringstream ss;
    ss << patternSize.height << " rows by " << patternSize.width << " cols, checker size " << patternMetric.height << "-by-" << patternMetric.width << " unit2";

    return ss.str();
}

void CalibGraphBuilder::TargetDef::GetObjectPoints(Points3F& pts, Indices& corners) const
{
    pts.clear();
    pts.reserve(patternSize.area());

    corners.clear();
    corners.resize(4);

    // generating object points
    for (int i = 0; i < patternSize.height; i++)
    {
        for (int j = 0; j < patternSize.width; j++)
        {
            pts.push_back(cv::Point3f(
                (float)((j+1) * patternMetric.width),  // X
                (float)((i+1) * patternMetric.height), // Y
                0 // Z = 0 (since we have a planar target)
            ));

            if ((i == 0 /******************/ && /*****************/ j == 0) ||
                (i == patternSize.height - 1 && /*****************/ j == 0) ||
                (i == 0 /******************/ && j == patternSize.width - 1) ||
                (i == patternSize.height - 1 && j == patternSize.width - 1))
            {
                size_t idx = pts.size() - 1;
                corners.push_back(idx);
            }
        }
    }

    // swap last two corners so they are arranged clockwise
    assert(corners.size() == 4);
    Indices::iterator last = boost::prior(corners.end());
    std::iter_swap(last, boost::prior(last));
}

bool CalibGraphBuilder::ImageFileList::CheckPattern(const String& pattern)
{
    String test = pattern;
    return replace(test, "%1", "0") && replace(test, "%2", "0");
}

bool CalibGraphBuilder::ImageFileList::FromPattern(const String& pattern, size_t cams, size_t imgs)
{
    if (cams == 0 || imgs == 0 || !CheckPattern(pattern)) return false;

    m_list.clear();
    m_list.resize(cams);

    Strings digits(std::max(cams, imgs));
    for (size_t i = 0; i < digits.size(); i++)
    {
        std::stringstream ss; ss << i;
        digits[i] = ss.str();
    }

    for (size_t cam = 0; cam < cams; cam++)
    {
        String imagePattern = pattern;
        std::stringstream ss;
        replace(imagePattern, "%1", digits[cam]);

        m_list[cam].resize(imgs);

        for (size_t img = 0; img < imgs; img++)
        {
            String filename = imagePattern;
            bool   success  = replace(filename, "%2", digits[img]);

            if (!success)
            {
                m_list.clear(); // this should make no difference
                E_ERROR << "error replacing image index in \"" << filename << "\"";

                return false;
            }

            m_list[cam][img] = filename;
        }
    }

    return true;
}

bool CalibGraphBuilder::ImageFileList::FromFile(const Path& path, size_t cams, size_t imgs)
{
    std::ifstream f(path.string().c_str());
    String line;

    if (!f.is_open())
    {
        E_ERROR << "error opening list file " << path;
        return false;
    }

    m_list.clear();
    m_list.resize(cams);

    for (size_t cam = 0; cam < m_list.size(); cam++)
    {
        m_list[cam].resize(imgs);

        for (size_t img = 0; img < imgs; img++)
        {
            if (!getline(f, line))
            {
                m_list.clear(); // this should make no difference

                E_ERROR << "error parsing list file " << path;
                E_ERROR << "reached EOF while trying to read image " << img << " of camera " << cam;

                return false;
            }

            m_list[cam][img] = line;
        }
    }

    return true;
}

bool CalibGraphBuilder::ImageFileList::CheckExists() const
{
    if (!IsOkay())
    {
        E_WARNING << "image file check aborted due to empty file list";
        return false;
    }

    size_t cams = m_list.size();
    size_t imgs = m_list.begin()->size();
    size_t checked = 0;

    // consistency check
    for (size_t cam = 1; cam < cams; cam++)
    {
        if (m_list[cam].size() != imgs)
        {
            E_WARNING << "image file check aborted due to inconsistent number of image";
            return false;
        }
    }

    E_INFO << "start checking image file..";

    for (size_t img = 0; img < imgs; img++)
    {
        for (size_t cam = 0; cam < cams; cam++)
        {
            Path imagePath = (*this)(cam, img);
            //cv::Mat im = cv::imread(imagePath.string());

            if (/*im.empty()*/ !fileExists(imagePath))
            {
                E_ERROR << "error reading image from " << imagePath;
                return false;
            }

            E_INFO << "checked " << imagePath << " (camera " << cam << ", image " << img << ")";
            checked++;
        }
    }

    E_INFO << checked << " image file(s) checked";

    return true;
}

bool CalibGraphBuilder::SetFileList(const String& filelist)
{
    if (!m_imageFiles.FromPattern(filelist, m_cams, m_views) &&
        !m_imageFiles.FromFile   (filelist, m_cams, m_views))
    {
        E_ERROR << "cannot decide image file list from \"" << filelist << "\"";
        return false;
    }

    if (!m_imageFiles.IsOkay())
    {
        E_ERROR << "profile creation failed due to invalid image file list";
        return false;
    }

    if (!m_imageFiles.CheckExists())
    {
        E_ERROR << "profile creation failed due to at least one missing file";
        return false;
    }

    return true;
}

void CalibGraphBuilder::SetFlag(int flag, bool enable)
{
    if (enable) m_flag |= flag;
    else m_flag &= ~flag;
}

bool CalibGraphBuilder::Build(CalibGraph& graph) const
{
    if (!graph.Create(m_cams, m_views))
    {
        E_ERROR "error initialising calibration graph";
        return false;
    }

    std::cout << "Start building calibration data from images.." << std::endl;
    int vw = (int) std::ceil(std::log10(m_views));
    const cv::Size subpxZeroZone(-1, -1);
    bool subpixel = m_subpxWinSize.area() > 0;
 
    Points3F objectPoints;
    Indices cornersIdx;
    m_targetDef.GetObjectPoints(objectPoints, cornersIdx);

    for (size_t view = 0; view < m_views; view++)
    {
        std::cout << "View " << std::setw(vw) << (view + 1) << "/" << m_views << "..";

        CalibGraph::ViewVertex::Ptr vxview = graph.m_viewvtx[view];
        size_t hits = 0;

        for (size_t cam = 0; cam < m_cams; cam++)
        {
            CalibGraph::CameraVertex::Ptr vxcam = graph.m_camvtx[cam];
            CalibGraph::Observation& o = graph.GetObservation(cam, view);

            o.source = m_imageFiles(cam, view);
            cv::Mat im = cv::imread(o.source.string(), cv::IMREAD_GRAYSCALE);

            if (im.empty())
            {
                std::cout << "..x (cam=" << cam << "" << std::endl;
                E_ERROR << "error loading image " << o.source;
                
                return false;
            }

            // initialise camera upon the first observation from it
            if (!vxcam->initialised)
            {
                assert(vxcam->imageSize.area() == 0);

                vxcam->SetIndex(cam);
                vxcam->imageSize = im.size();
                vxcam->initialised = true;
            }
            else if (vxcam->imageSize != im.size())
            {
                E_ERROR << "expected image size of camera " << cam << " is " << size2string(vxcam->imageSize);
                E_ERROR << "while the image loaded from " << o.source << " has " << size2string(im.size());

                return false;
            }

            bool found = cv::findChessboardCorners(im, m_targetDef.patternSize, o.imagePoints, m_flag);

            if (!found)
            {
                std::cout << "..x";
                continue;
            }

            if (subpixel) // refine found corners
            {
                cv::cornerSubPix(im, o.imagePoints, m_subpxWinSize, subpxZeroZone, m_subpxTerm);
            }

            // initialise view upon its first observation
            if (!vxview->initialised)
            {
                vxview->SetIndex(view);
                vxview->objectPoints = objectPoints;
                vxview->initialised = true;
            }

            o.SetActive(true);
            hits++;

            std::cout << ".." << cam;
         }
        std::cout << (hits > 0 ? (hits == m_cams ? "..PERFECT" : "..GOOD") : "..REJECTED") << std::endl;
    }

    return true;
}
