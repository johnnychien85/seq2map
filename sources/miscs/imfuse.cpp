#include <seq2map/common.hpp>

using namespace seq2map;

struct UVMap
{
	cv::Mat umap;
	cv::Mat vmap;
};

typedef std::vector<UVMap> UVMaps;

bool parseArgs(int, char*[], UVMaps& maps, Paths& rawPaths);



int main(int argc, char* argv[])
{
	Paths rawPaths;
	UVMaps maps;

	if (!parseArgs(argc, argv, maps, rawPaths)) return EXIT_FAILURE;

	try
	{
		namespace fs = boost::filesystem;
		Paths files = enumerateFiles(rawPaths[0]);

		if (files.empty())
		{
			E_INFO << "no images found";
			return EXIT_SUCCESS;
		}

		std::sort(files.begin(), files.end());

		int key = 0;
		size_t i = 0;
		size_t shift = 0;
		const size_t step = 32;
		bool drawEpipolarLines = false;
		bool equalisation = false;

		while ((key = cv::waitKey(0)) != 'q')
		{
			switch (key)
			{
			case 'a': i -= i > 0 ? 1 : 0;                break;
			case 'd': i += i < files.size() - 1 ? 1 : 0; break;
			case 'w': shift = shift == 0 ? step - 1 : shift - 1;  break;
			case 's': shift = shift == step - 1 ? 0 : shift + 1; break;
			case ' ': drawEpipolarLines = !drawEpipolarLines; break;
			case 'e': equalisation      = !equalisation;      break;
			}

			std::vector<cv::Mat> im(maps.size());

			for (size_t k = 0; k < maps.size(); k++)
			{
				Path file = rawPaths[k] / files[i].filename();
				im[k] = cv::imread(file.string(), cv::IMREAD_GRAYSCALE);

				if (im[k].empty())
				{
					E_ERROR << "error loading image " << file;
					return EXIT_FAILURE;
				}

				cv::remap(im[k], im[k], maps[k].umap, maps[k].vmap, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar());

				if (equalisation) cv::equalizeHist(im[k], im[k]);
			}

			cv::Mat canvas;

            switch (maps.size())
            {
            case 1:
                canvas = gray2rgb(im[0]);
                break;
            case 2:
                canvas = imfuse(im[0], im[1]);
                break;
            case 3:
                cv::merge(im, canvas);
                break;
            }

			if (drawEpipolarLines)
			{
				for (size_t y = shift; y < canvas.rows; y += step)
				{
					cv::Point pt1(0, y);
					cv::Point pt2(canvas.cols, y);
					cv::line(canvas, pt1, pt2, cv::Scalar(0, 255, 0), 1);
				}
			}

			cv::imshow("Fusion", canvas);
		}
	}
	catch (std::exception& ex)
	{
		E_ERROR << "error fusing image";
        E_ERROR << ex.what();

		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

bool parseArgs(int argc, char* argv[], UVMaps& maps, Paths& rawPaths)
{
	if (argc != 3)
	{
		E_ERROR << "FUUUUUUUUUUUUUUUUCK";
		return false;
	}

    Path calPath = argv[1];
	Path rawPath = argv[2];

	Paths calPaths = enumerateFiles(calPath);
	rawPaths = enumerateDirs(rawPath);

	if (rawPaths.size() > 3 || calPaths.size() != rawPaths.size())
	{
        E_ERROR << "raw path(s) in " << rawPath;
        for (size_t i = 0; i < rawPaths.size(); i++) E_INFO << rawPaths[i];

        E_ERROR << "cal path(s) in " << calPath;
        for (size_t i = 0; i < calPaths.size(); i++) E_INFO << calPaths[i];

		E_ERROR << "SHIT!";

		return false;
	}

	maps.resize(calPaths.size());
	std::sort(calPaths.begin(), calPaths.end());
	std::sort(rawPaths.begin(), rawPaths.end());

	for (size_t i = 0; i < calPaths.size(); i++)
	{
		cv::FileStorage fs(calPaths[i].string().c_str(), cv::FileStorage::READ);
		cv::Mat K, D, R_rect, P_rect;
		cv::Size S_rect;

		if (!fs.isOpened())
		{
			E_ERROR << "error reading profile " << calPaths[i];
			return false;
		}

		try
		{
			fs["K"] >> K;
			fs["D"] >> D;
			fs["S_rect"] >> S_rect;
			fs["R_rect"] >> R_rect;
			fs["P_rect"] >> P_rect;

			// sanity check
			if (S_rect.area() == 0 || K.empty() || D.empty() || R_rect.empty() || P_rect.empty())
			{
				E_ERROR << "some of parameters including K, D, R_rect, P_rect and S_rect are missing";
				return false;
			}
		}
		catch (std::exception& ex)
		{
			E_ERROR << "error parsing camera parameters from " << calPaths[i];
			E_ERROR << ex.what();

			return false;
		}

		cv::initUndistortRectifyMap(K, D, R_rect, P_rect, S_rect, CV_16SC2, maps[i].umap, maps[i].vmap);

		E_INFO << calPaths[i] << " parsed";
	}

	return true;
}

