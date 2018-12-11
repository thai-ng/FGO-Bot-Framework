#include <array>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <chrono>
#include <string_view>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

constexpr int ResizeFactor = 2;
constexpr int MarginFactor = 160;
constexpr double ServantWidthFactor = 1 / 4.0;

constexpr double ClassWidthFactor = 7 / 20.0;
constexpr double ClassHeightFactor = 2 / 7.0;

enum class ServantClass : int {
	Saber = 0,
	Archer,
	Lancer,
	Rider,
	Assassin,
	Caster,
	Berserker,
	Avenger,
	AlterEgo,
	MoonCancer,
	Foreigner,
	Shielder,
	Unknown
};

bool CheckTemplate(cv::Mat& image, cv::Mat const& templ) {
	auto resultCols = image.cols - templ.cols + 1;
	auto resultRows = image.rows - templ.rows + 1;
	cv::Mat result{ cv::Size(resultCols, resultRows), CV_32FC1 };
	cv::matchTemplate(image, templ, result, cv::TemplateMatchModes::TM_CCOEFF_NORMED);

	double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

	if (maxVal > 0.9) {
		cv::Rect resultRect{ maxLoc, cv::Point{maxLoc.x + templ.cols, maxLoc.y + templ.rows} };
		cv::rectangle(image, resultRect, cv::Scalar{ 0, 0, 255 });
		return true;
	}

	return false;
}

cv::Mat OpenTemplate(std::string_view templateName) {
	auto templ = cv::imread(templateName.data());
	cv::resize(templ, templ, cv::Size(templ.cols / ResizeFactor, templ.rows / ResizeFactor));
	return templ;
}

template <typename T>
constexpr auto BaseVal(T t) {
	return static_cast<typename std::underlying_type_t<T>>(t);
}

class Servant{
public:
	Servant() {}

	Servant(cv::Mat inputMat, std::array<cv::Mat, 3> const& templates) :
		sourceImage(std::move(inputMat)),
		servantClass(GetServantClass(*this, templates))
	{
	}

	void drawBorder() {
		cv::Rect border{ 0, 0, sourceImage.cols, sourceImage.rows };
		auto redBrush = cv::Scalar{ 0, 0, 255 };
		cv::rectangle(sourceImage, border, redBrush);
	}

	void drawClassArea() {
		auto classRect = getClassRect();
		auto redBrush = cv::Scalar{ 0, 0, 255 };
		cv::rectangle(sourceImage, classRect, redBrush);
	}

	static ServantClass GetServantClass(Servant const& servant, std::array<cv::Mat, 3> const& templates) {
		auto classArea = servant.getClassArea();

		// TODO(Thai): Figure out how to tie this to the array. 
		// Maybe build the image along with the binary for constexpr array?
		for (auto i = 0; i < BaseVal(ServantClass::Unknown); ++i) {
			if (CheckTemplate(classArea, templates[i]))
				return static_cast<ServantClass>(i);
		}

		return ServantClass::Unknown;
	}

	ServantClass Class() const {
		return servantClass;
	}

private:
	cv::Rect getClassRect() const {
		auto classHeight = (sourceImage.rows / 7) * 2;
		auto classWidth = (sourceImage.cols / 20) * 7;
		auto classX = 0;
		auto classY = sourceImage.rows - classHeight;
		cv::Rect classRect{ classX, classY, classWidth, classHeight };

		return classRect;
	}

	cv::Mat getClassArea() const {
		auto classRect = getClassRect();
		return cv::Mat(sourceImage, classRect);
	}

	cv::Mat sourceImage;
	ServantClass servantClass;
};

int main() {
	auto scene = cv::imread("scene.jpg");
	cv::resize(scene, scene, cv::Size(scene.cols / ResizeFactor, scene.rows / ResizeFactor));
	std::array<cv::Mat, 3> templates =
	{
		OpenTemplate("templates/caster.png"),
		OpenTemplate("templates/archer.png"),
		OpenTemplate("templates/lancer.png")
	};

	auto startTime = std::chrono::steady_clock::now();
	auto sceneWidth = scene.cols;

	auto margin = scene.cols / MarginFactor;
	auto servantWidth = static_cast<int>(sceneWidth * ServantWidthFactor);

	auto midHeight = scene.rows / 2;
	auto height = scene.rows;

	std::array<Servant, 3> servants;
	int currentServantIndex = 0;
	auto heightRange = cv::Range(midHeight, height);
	std::generate(std::begin(servants), std::end(servants), [&] {
		cv::Rect servantRect{ margin + (currentServantIndex * servantWidth), midHeight, servantWidth, scene.rows / 2 };
		++currentServantIndex;
		return Servant(cv::Mat(scene, servantRect), templates);
	});

	auto endTime = std::chrono::steady_clock::now();
	
	for (auto& servant : servants) {
		auto servantClass = servant.Class();
		servant.drawBorder();
		servant.drawClassArea();
		if (servantClass == ServantClass::Caster) {
			std::cout << "Class: Caster\n";
		}
		else if (servantClass == ServantClass::Lancer) {
			std::cout << "Class: Lancer\n";
		}
		else if (servantClass == ServantClass::Archer) {
			std::cout << "Class: Archer\n";
		}
		else {
			std::cout << "Class: Unknown\n";
		}
	}

	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	std::cout << "Took " << elapsed.count() << "ms\n";
	
	cv::namedWindow("scene");
	cv::imshow("scene", scene);

	cv::waitKey(0);
	return 0;
}