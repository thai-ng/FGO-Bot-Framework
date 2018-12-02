#include <array>
#include <algorithm>
#include <sstream>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

constexpr int ResizeFactor = 2;
constexpr int MarginFactor = 160;
constexpr double ServantWidthFactor = 1 / 4.0;

constexpr double ClassWidthFactor = 7 / 20.0;
constexpr double ClassHeightFactor = 2 / 7.0;

enum class ServantClass {
	Saber,
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

class Servant{
public:
	Servant() {}

	Servant(cv::Mat inputMat, std::string n) :
		name(std::move(n)),
		sourceImage(std::move(inputMat)),
		servantClass(GetServantClass(*this))
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

	void show() const {
		cv::namedWindow(name);
		cv::imshow(name, sourceImage);
	}

	static ServantClass GetServantClass(Servant const& servant) {
		auto classArea = servant.getClassArea();

		if (CheckTemplate(classArea, OpenTemplate("templates/caster.png")))
			return ServantClass::Caster;
		else if (CheckTemplate(classArea, OpenTemplate("templates/archer.png")))
			return ServantClass::Archer;
		else if (CheckTemplate(classArea, OpenTemplate("templates/lancer.png")))
			return ServantClass::Lancer;
		else
			return ServantClass::Unknown;
	}

	ServantClass Class() const {
		return servantClass;
	}

	std::string const& Name() const {
		return name;
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

	std::string name;
	cv::Mat sourceImage;
	ServantClass servantClass;
};

int main() {
	auto scene = cv::imread("scene.jpg");
	cv::resize(scene, scene, cv::Size(scene.cols / ResizeFactor, scene.rows / ResizeFactor));
	auto sceneWidth = scene.cols;

	auto margin = scene.cols / MarginFactor;
	auto servantWidth = static_cast<int>(sceneWidth * ServantWidthFactor);
	
	auto midHeight = scene.rows / 2;
	auto height = scene.rows;

	std::array<Servant, 3> servants;
	int currentServantIndex = 0;
	auto heightRange = cv::Range(midHeight, height);
	std::generate(std::begin(servants), std::end(servants), [&] {
		std::stringstream str;
		str << "Servant " << currentServantIndex + 1;
		cv::Rect servantRect{ margin + (currentServantIndex * servantWidth), midHeight, servantWidth, scene.rows / 2 };
		++currentServantIndex;
		return Servant(cv::Mat(scene, servantRect), str.str());
	});
	
	for (auto& servant : servants) {
		auto servantClass = servant.Class();
		servant.drawBorder();
		servant.drawClassArea();
		if (servantClass == ServantClass::Caster) {
			std::cout << servant.Name() << " Class: Caster\n";
		}
		else if (servantClass == ServantClass::Lancer) {
			std::cout << servant.Name() << " Class: Lancer\n";
		}
		else if (servantClass == ServantClass::Archer) {
			std::cout << servant.Name() << " Class: Archer\n";
		}
		else {
			std::cout << servant.Name() << " Class: Unknown\n";
		}
	}

	cv::namedWindow("scene");
	cv::imshow("scene", scene);

	cv::waitKey(0);
	return 0;
}