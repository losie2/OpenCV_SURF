#include <iostream>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <math.h>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

constexpr auto PI = 3.141592;;

const int LOOP_NUM = 10;
const int GOOD_PTS_MAX = 50;
const float GOOD_PORTION = 0.15f;

int64 work_begin = 0;
int64 work_end = 0;

/*
	게임 플레이
*/
int gameTurns = 2;
int turn = 0;

struct Player
{
	int hp = 30;
	int damage = 10;
};

/*
    뷰 영상 출력을 위한 구조체
*/
struct Axis
{
    int x;
    int y;
    int z;
};
struct AxisDouble
{
    double x;
    double y;
    double z;
};

Mat imgView;
Mat imgPanorama;
double HorizontalViewAngle = 1.5f;
double VerticalViewAngle = 1.2f;
int ViewSize = 500;

Mat inverse;
std::vector<Point2f> inverse_corner;

double yAxisRotate = 0;
double zAxisRotate = 0;

vector<Mat> cubemapface[6]; // Back, Left, Front, Right, Top, Bottom
int faceIndex;
int indexX, indexY;

Axis getNearPixel(Mat Img, int x, int y);
Mat remapImage(Mat imgIn);
AxisDouble CalcThetaPhiToXYZ(double theta, double phi);
AxisDouble CalcCubicXYZ(double x, double y, double z);
void GenView(Mat SourceImg);
void KeyDownEvent(int ch, Mat SourceImg);

/*
	Image Interpolation
*/
Axis getNearPixel(Mat Img, int x, int y)
{
	int col = 1;
	int row = 1;
	bool done = false;
	Axis xy;

	while (!done)
	{
		for (int i = -col; i <= col; i++)
		{
			for (int j = -row; j <= row; j++)
			{
				if (x + i <= 0 || y + j <= 0)
				{
					continue;
				}
				if (x + i >= Img.cols || y + j >= Img.rows)
				{
					continue;
				}
				// (3, 3) 범위 내에서 3채널에 모두 보간.
				if (!(Img.at<Vec3b>(Point(x + i, y + j))[0] == 0 && Img.at<Vec3b>(Point(x + i, y + j))[1] == 0 && Img.at<Vec3b>(Point(x + i, y + j))[2] == 0))
				{
					xy.x = x + i;
					xy.y = y + j;
					return xy;
				}
			}
		}
		col++;
		row++;

		if (col > 3 || row > 3)
		{
			xy.x = x;
			xy.y = y;
			return xy;
		}
	}
}
Mat remapImage(Mat imgIn)
{
	int width = imgIn.cols;
	int height = imgIn.rows;
	Mat RemapImg = Mat(height, width, CV_8UC3, Scalar(0, 0, 0));

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			Axis xy = getNearPixel(imgIn, i, j);
			RemapImg.at<Vec3b>(Point(i, j)) = imgIn.at<Vec3b>(Point(xy.x, xy.y));
		}
	}

	return RemapImg;

}

/*
	Generate View
*/
void GenView(Mat SourceImg)
{
	int width = SourceImg.cols;
	int height = SourceImg.rows;

	int Multiple = 2; // 연산의 속도를 위함
	double radius = height / 2 / Multiple; // CubemapEdgeLength / 2, Normalization => -1 ~ 0, 0 ~ 1

	int viewWidth = (int)((double)height * HorizontalViewAngle / Multiple);
	int viewHeight = (int)((double)height * VerticalViewAngle / Multiple);

	imgView = Mat(viewHeight, viewWidth, CV_8UC3, Scalar(0, 0, 0));
	cv::resize(SourceImg, imgPanorama, Size(width, height), 0, 1);


	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			Vec3b pixel = SourceImg.at<Vec3b>(Point(i, j));

			double theta = (((double)j / height) * PI) - (PI * 0.5f);	//-pi/2 ~ pi/2
			double phi = (((double)i / width) * 2 * PI) - PI;			//-pi ~ pi

			AxisDouble axis = CalcThetaPhiToXYZ(theta, phi);

			if (axis.x > 0)
			{
				/*
					전체 구면 좌표계 내에서 x > 0 인 평면만 고려,
					회전 변환을 거친 후 좌표계에는 변동이 없다는 점을 이용.
				*/
				AxisDouble axis2;
				axis2 = CalcCubicXYZ(axis.x, axis.y, axis.z);

				/*
					Transformation normalization Cubemap Coordinate to Image Viewer Pixel Coordinate
					Length of edge on YZ-Face

				*/
				int y = (int)(axis2.y * radius) + viewWidth / 2;
				int z = (int)(axis2.z * radius) + viewHeight / 2;

				if (y >= 0 && y < viewWidth && z >= 0 && z < viewHeight)
				{

					int PointX = y;
					int PointY = z;

					if (PointX < 5 || PointX > viewWidth - 6 || PointY < 5 || PointY > viewHeight - 6)
					{
						Vec3b C = { 255, 0, 0 }; // BGR
						imgPanorama.at<Vec3b>(Point(i, j)) = C;
					}

					imgView.at<Vec3b>(Point(PointX, PointY)) = pixel;

				}
			}
		}
	}
	imgView = remapImage(imgView);
	cv::resize(imgView, imgView, Size(ViewSize, (int)((double)ViewSize / (HorizontalViewAngle / VerticalViewAngle))), 0, 1);
	cv::imshow("Panorama", imgPanorama);
	cv::imshow("View", imgView);

	int ch = waitKey();
	KeyDownEvent(ch, SourceImg);
}
AxisDouble CalcThetaPhiToXYZ(double theta, double phi)
{
	AxisDouble axis;

	//theta, phi -> x,y,z 변환
	double x1 = cos(theta) * cos(phi);
	double y1 = cos(theta) * sin(phi);
	double z1 = sin(theta);

	//z축 회전 변환
	double x2 = x1 * cos(zAxisRotate) - y1 * sin(zAxisRotate);
	double y2 = x1 * sin(zAxisRotate) + y1 * cos(zAxisRotate);
	double z2 = z1;

	//y축 회전 변환
	double x3 = x2 * cos(yAxisRotate) + z2 * sin(yAxisRotate);
	double y3 = y2;
	double z3 = -x2 * sin(yAxisRotate) + z2 * cos(yAxisRotate);

	axis.x = x3;
	axis.y = y3;
	axis.z = z3;

	return axis;
}
void KeyDownEvent(int ch, Mat SourceImg)
{
	switch (ch)
	{
	case 's'://Theta Up
		//if (thetaPlus <= PI / 2)
	{
		yAxisRotate += 5 * (PI / 365);
		//if (thetaPlus > PI / 2)
		{
			//	thetaPlus = PI / 2;
		}
		cout << "Lat : " << -yAxisRotate << ", Long : " << -zAxisRotate << endl;
	}

	GenView(SourceImg);
	break;

	case 'w'://Theta Down
		//if (thetaPlus >= -PI / 2)
	{
		yAxisRotate -= 5 * (PI / 365);
		//if (thetaPlus < -PI / 2)
		{
			//	thetaPlus = -PI / 2;
		}
		cout << "Lat : " << -yAxisRotate << ", Long : " << -zAxisRotate << endl;
	}

	GenView(SourceImg);
	break;

	case 'd'://Phi Down
		zAxisRotate -= 5 * (2 * PI / 365);

		if (zAxisRotate <= 0)
		{
			zAxisRotate = 2 * PI;
		}
		cout << "Lat : " << -yAxisRotate << ", Long : " << -zAxisRotate << endl;

		GenView(SourceImg);
		break;

	case 'a'://Phi Up
		zAxisRotate += 5 * (2 * PI / 365);
		if (zAxisRotate >= 2 * PI)
		{
			zAxisRotate = 0;
		}
		cout << "Lat : " << -yAxisRotate << ", Long : " << -zAxisRotate << endl;

		GenView(SourceImg);
		break;

	default:
		GenView(SourceImg);
		break;
	}
}
AxisDouble CalcCubicXYZ(double x, double y, double z)
{
	AxisDouble axis;

	/*
	큐브맵화, 항상 Front Face에서 렌더링 하기 때문에 단위벡터는 x이다. .
	*/
	double a = x;
	double x1 = x / a;
	double y1 = y / a;
	double z1 = z / a;

	axis.x = x1;
	axis.y = y1;
	axis.z = z1;

	return axis;
}

/*
	1. Spherical -> Cubemap
	2. Cubemap -> Spherical
*/
float* GetCubemapCoordinate(int x, int y, int face, int edge, float* point)
{
	/*
		-1 ~ 1 사이 정규화된 좌표값 내에서 point를 정해주어야 한다.
		즉, 2의 길이가 필요하기 때문에 a, b에 2를 곱해준다.
		edge < y < 2*edge 사이 b = 2 * 1.??????f이고, 2~4의 범위를 갖는다.
	*/
	float a = 2 * (x / (float)edge);
	float b = 2 * (y / (float)edge);

	if (face == 0) { point[0] = -1; point[1] = 1 - a; point[2] = 3 - b; }         // Back
	else if (face == 1) { point[0] = a - 3, point[1] = -1;     point[2] = 3 - b; }   // Left
	else if (face == 2) { point[0] = 1;      point[1] = a - 5; point[2] = 3 - b; }   // Front
	else if (face == 3) { point[0] = 7 - a;   point[1] = 1;     point[2] = 3 - b; }   // Right
	else if (face == 4) { point[0] = a - 3;   point[1] = 1 - b; point[2] = 1; }   // Top
	else if (face == 5) { point[0] = a - 3;   point[1] = b - 5; point[2] = -1; }   // Bottom

	return point;
}
Mat CvtCub2Sph(Mat* cube, Mat* original) {
	/*
		구면 파노라마 이미지의 길이는 큐브 맵의 길이를 따른다.
		구면 파노라마 이미지의 높이는 큐브 맵 길이의 1/2.

		1. 구면 파노라마 이미지 좌표(j, i)를 정규화, 구면 좌표계로 변환한다.
		2. 구면 좌표계에 대응하는 큐브맵의 좌표를 찾는다.
		3. 찾은 큐브맵의 좌표가 어느 face에 있는지 찾는다.
		4. 큐브맵의 좌표를 구면 파노라마 이미지의 좌표(j, i)에 대입한다.
	*/



	int Width = cube->size().width;
	float Height = (0.5f) * cube->size().width;

	Mat spherical;
	spherical = Mat::zeros(original->size().height, original->size().width, cube->type());

	/*
		좌표계를 0부터 1로 정규화 한다. (0, 0)
		경도를 나타내기 위한 변수 phi
		위도를 나타내기 위한 변수 theta
	*/
	float u, v;
	float phi, theta;
	int cubeFaceWidth, cubeFaceHeight;

	cubeFaceWidth = cube->size().width / 4;
	cubeFaceHeight = cube->size().height / 3;

	for (int i = 0; i < 6; i++)
	{
		cubemapface->push_back(Mat::zeros(cubeFaceWidth, cubeFaceHeight, cube->type()));
	}
	for (int j = 0; j < Height; j++)
	{
		/*
			(i = 0, j = 0) 부터 j를 높이까지 증가.
			즉, 구면의 위도 생성.
			왼쪽 아래부터 시작.
		*/
		v = 1 - ((float)j / Height);
		theta = v * PI;

		for (int i = 0; i < Width; i++)
		{
			// 위도 상의 한 점(0부터)에서 경도 끝까지 증가
			u = ((float)i / Width);
			phi = u * 2 * PI;

			float x, y, z; // 단위 벡터
			x = cos(phi) * sin(theta) * -1;
			y = sin(phi) * sin(theta) * -1;
			z = cos(theta);

			float xa, ya, za;
			float a;

			a = max(abs(x), max(abs(y), abs(z)));

			/*
				큐브 면 중 하나에 있는 단위 벡터와 평행한 벡터.
				이 때, ya가 -1인지 1인지(Left, Right) 값을 보고 평면을 결정.

				ya가 1 or -1이라면 y벡터의 변화가 없다는 뜻. 즉 xz평면만 고려한다는 의미.
				xa와 za도 동일하게 적용.
			*/
			xa = x / a;
			ya = y / a;
			za = z / a;

			int xPixel, yPixel;
			int xOffset, yOffset;


			/*
				1. 정규화를 거친 좌표계이기 때문에 2.f로 나누어준다.
				2. -1 ~ 1로 정규화 되어있는 좌표계에 edge 길이(cubeFaceWidth, cubeFaceHeight)를 곱해준다.
				3. WorldOffset에 적용한다. 이때 Offset은 큐브맵 좌표계에 따른다.
			*/
			if (ya == -1)
			{
				//Left
				xPixel = (int)((((xa + 1.f) / 2.f)) * cubeFaceWidth);
				xOffset = cubeFaceWidth;
				yPixel = (int)((((za + 1.f) / 2.f)) * cubeFaceHeight);
				yOffset = cubeFaceHeight;
				indexX = abs(xPixel);
				indexY = abs(yPixel);
				faceIndex = 1;
			}
			else if (ya == 1)
			{
				//Right
				xPixel = (int)((((xa - 1.f) / 2.f)) * cubeFaceWidth);
				xOffset = 3 * cubeFaceWidth;
				yPixel = (int)((((za + 1.f) / 2.f)) * cubeFaceHeight);
				yOffset = cubeFaceHeight;
				indexX = abs(xPixel);
				indexY = abs(yPixel);
				faceIndex = 3;
			}
			else if (za == -1)
			{
				//Top
				xPixel = (int)((((xa + 1.f) / 2.f)) * cubeFaceWidth);
				xOffset = cubeFaceWidth;
				yPixel = (int)((((ya - 1.f) / 2.f)) * cubeFaceHeight);
				yOffset = 0;
				indexX = abs(xPixel);
				indexY = abs(yPixel);
				faceIndex = 4;
			}
			else if (za == 1)
			{
				//Bottom
				xPixel = (int)((((xa + 1.f) / 2.f)) * cubeFaceWidth);
				xOffset = cubeFaceWidth;
				yPixel = (int)((((ya + 1.f) / 2.f)) * cubeFaceHeight);
				yOffset = 2 * cubeFaceHeight;
				indexX = abs(xPixel);
				indexY = abs(yPixel);
				faceIndex = 5;
			}
			else if (xa == 1)
			{
				//Front
				xPixel = (int)((((ya + 1.f) / 2.f)) * cubeFaceWidth);
				xOffset = 2 * cubeFaceWidth;
				yPixel = (int)((((za + 1.f) / 2.f)) * cubeFaceHeight);
				yOffset = cubeFaceHeight;
				indexX = abs(xPixel);
				indexY = abs(yPixel);
				faceIndex = 2;
			}
			else if (xa == -1)
			{
				//Back
				xPixel = (int)((((ya - 1.f) / 2.f)) * cubeFaceWidth);
				xOffset = 0;
				yPixel = (int)((((za + 1.f) / 2.f)) * cubeFaceHeight);
				yOffset = cubeFaceHeight;
				indexX = abs(xPixel);
				indexY = abs(yPixel);
				faceIndex = 0;
			}
			else
			{
				xPixel = 0;
				yPixel = 0;
				xOffset = 0;
				yOffset = 0;
			}
			xPixel = abs(xPixel);
			yPixel = abs(yPixel);

			xPixel += xOffset;
			yPixel += yOffset;

			//cout << faceIndex << " " << indexX << " " << indexY << " " << '\n';

			if (indexX != cubeFaceWidth && indexY != cubeFaceHeight) { // 큐브맵 이미지 out of bound 방지
				cubemapface->at(faceIndex).at<Vec3b>(indexY, indexX) = cube->at<Vec3b>(yPixel, xPixel);
				spherical.at<Vec3b>(j, i) = cube->at<Vec3b>(yPixel, xPixel);
			}
		}

	}
	for (int i = 0; i < 6; i++)
	{
		cubemapface->at(i) = remapImage(cubemapface->at(i));
	}
	spherical = remapImage(spherical);
	return spherical;
}
Mat CvtSph2Cub(Mat* pano) {

    /*
       구면 파노라마 이미지 너비, 높이 구하기
       이 때 cubeWidth는 6칸(Back, Left, Front, Right, Top, Bottom)으로 나누어진 맵의 가로값. 파노라마 이미지의 가로값과 같음
       cubeHeight는 맵의 세로값. 즉 (Top, Left, Bottom) 세 칸을 차지함.
       본 이미지는 1:2 비율을 갖고 4칸:3칸이므로 0.75f를 곱해줌.
    */
    int cubeWidth = pano->size().width;
    float cubeHeight = (0.75f) * pano->size().width;

    /*
        edge(한 큐브맵의 선)는 정사각형의 한 선이므로 width / 4
    */
    int edge = pano->size().width / 4;
    int face, startIndex, range;


    /*
        Mat은 col, row를 사용.
    */
    Mat cubemap;
    cubemap = Mat::zeros(cubeHeight, cubeWidth, pano->type());

    /*
        가로(cubeWidth)는 동일한 상태에서
        세로3칸 가로4칸을 수행하기 위해 face가 1(가로) 혹은 4,5(세로) 일 때의 경우 범위를 다르게 접근.
    */
    for (int x = 0; x < cubeWidth; x++)
    {
        face = x / edge;
        /*
            Left(face = 1)일 경우. 가로 범위이기 때문에 top, bottom은 고려할 필요가 없음
            face = 0, (1, 4, 5,) 2, 3 으로 진행.
        */
        if (face == 1) // Left(Top, Bottom)
        {
            startIndex = 0;
            range = 3 * edge;
        }
        else
        {
            startIndex = edge;
            range = 2 * edge;
        }

        /* face가 1일 때(Left) 아래 조건에 걸려 face가 4, 5로 변경되는 것을 막아주기 위한 변수. */
        int prev_face = face;

        for (int y = startIndex; y < range; y++)
        {
            if (y < edge) // Top
                face = 4;
            else if (y >= 2 * edge) // Bottom
                face = 5;

            /*
                1. 구면 파노라마 이미지 좌표를 큐브맵에 배치
                2. edge 값에 따라 face가 결정. face에 따른 3차원 좌표값(구면 파노라마 이미지 좌표계)을 큐브맵 좌표계(2차원)으로 투영.
                3. 투영된 2차원 좌표값을 구면 좌표계로 변환
                4. 변환된 구면 좌표계를 구면 파노라마 이미지의 좌표값으로 배치.
            */
            float* point = new float[3];
            point = GetCubemapCoordinate(x, y, face, edge, point);

            // 경도값
            float phi;

            // 위도값
            float theta;

            // 큐브맵 좌표계
            int polarX;
            int polarY;


            /*
                atan2(a, b)는 원점에서 (a, b) 까지 상대적인 각도(위치)이다.
                phi는 경도이고, 원점에서 (point[1] = y, point[0] = x)까지의 상대적인 각도를 구한다.
                이 때 (point[1] = y, point[0] = x)는 큐브맵 위의 한 face 위 점이므로 구면 파노라마의 한 좌표로 표현하기 위해 atan2를 사용한다.
                theta는 위도이고, 90 ~ -90 degree의 범위를 가진다.
                범위를 벗어나게 하지 않기 위해 x, y 사이 거리를 2번째 인자로 준다.
                원점이 기준이므로 원점과 (point[2] = z, 거리) 사이의 각도이다.
            */
            phi = atan2(point[1], point[0]);
            theta = atan2(point[2], sqrt(pow(point[0], 2) + pow(point[1], 2)));


            /*
                (phi + PI) / PI는 phi 만큼 회전한 구면 위의 이미지 상 (x, y)좌표이다.
                ((PI / 2) - theta) / PI는 90 ~ -90 범위내에서 theta만큼 회전한 구면 위의 이미지 상 (x, y, z)좌표이다.
                이렇게 구한 구면 파노라마 이미지 상의 좌표를 큐브맵 이미지 좌표계에 대입한다.
            */
            polarX = 2 * edge * ((phi + PI) / PI);
            polarY = 2 * edge * (((PI / 2) - theta) / PI);

			if(polarX != pano->size().width && polarY != pano->size().height) // 구면 이미지 out of bound 방지
				cubemap.at<Vec3b>(y, x) = pano->at<Vec3b>(polarY, polarX);

            face = prev_face;
        }
    }
	
	cubemap = remapImage(cubemap);
    return cubemap;
}

/*
	직선의 방정식
*/

float left(int y, vector<Point2f>& corners)
{
	float m = (corners[3].y - corners[0].y) / (corners[3].x - corners[0].x);
	float x = (1 / m) * (y - corners[3].y) + corners[3].x;
	return x;

}
float right(int y, vector<Point2f>& corners)
{
	float m = (corners[2].y - corners[1].y) / (corners[2].x - corners[1].x);
	float x = (1 / m) * (y - corners[2].y) + corners[2].x;
	return x;
}
float topEquation(int x, vector<Point2f>& corners)
{
	float m = (corners[1].y - corners[0].y) / (corners[1].x - corners[0].x);
	float y = m * (x - corners[1].x) + corners[1].y;
	return y;
}
float bottom(int x, vector<Point2f>& corners)
{
	float m = (corners[2].y - corners[3].y) / (corners[2].x - corners[3].x);
	float y = m * (x - corners[2].x) + corners[2].y;
	return y;
}

void imageShow(string name, int width, int height, InputArray& img)
{
	namedWindow(name, 0);
	resizeWindow(name, width, height);
	imshow(name, img);
}

/*
	SURF
*/
static void workBegin()
{
	work_begin = getTickCount();
}
static void workEnd()
{
	work_end = getTickCount() - work_begin;
}
static double getTime()
{
	return work_end / ((double)getTickFrequency()) * 1000.;
}
struct SURFDetector
{
	Ptr<Feature2D> surf;
	SURFDetector(double hessian = 800.0)
	{
		surf = SURF::create(hessian);
	}
	template<class T>
	void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
	{
		surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
	}
};

template<class KPMatcher>
struct SURFMatcher
{
	KPMatcher matcher;
	template<class T>
	void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
	{
		matcher.match(in1, in2, matches);
	}
};

static Mat drawGoodMatches(
	const Mat& img1,
	const Mat& img2,
	const std::vector<KeyPoint>& keypoints1,
	const std::vector<KeyPoint>& keypoints2,
	std::vector<DMatch>& matches,
	std::vector<Point2f>& scene_corners_
)
{
	//-- Sort matches and preserve top 10% matches
	std::sort(matches.begin(), matches.end());
	std::vector< DMatch > good_matches;
	double minDist = matches.front().distance;
	double maxDist = matches.back().distance;

	const int ptsPairs = std::min(GOOD_PTS_MAX, (int)(matches.size() * GOOD_PORTION));
	for (int i = 0; i < ptsPairs; i++)
	{
		good_matches.push_back(matches[i]);
	}
	std::cout << "\nMax distance: " << maxDist << std::endl;
	std::cout << "Min distance: " << minDist << std::endl;

	std::cout << "Calculating homography using " << ptsPairs << " point pairs." << std::endl;

	// drawing the results
	Mat img_matches;
	drawMatches(img1, keypoints1, img2, keypoints2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}
	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0, 0);
	obj_corners[1] = Point(img1.cols, 0);
	obj_corners[2] = Point(img1.cols, img1.rows);
	obj_corners[3] = Point(0, img1.rows);
	std::vector<Point2f> scene_corners(4);

	Mat H = findHomography(obj, scene, RANSAC);
	perspectiveTransform(obj_corners, scene_corners, H);

	inverse = H.inv();
	perspectiveTransform(scene_corners, obj_corners, inverse);

	inverse_corner = obj_corners;
	scene_corners_ = scene_corners;

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches,
		scene_corners[0] + Point2f((float)img1.cols, 0), scene_corners[1] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	line(img_matches,
		scene_corners[1] + Point2f((float)img1.cols, 0), scene_corners[2] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	line(img_matches,
		scene_corners[2] + Point2f((float)img1.cols, 0), scene_corners[3] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	line(img_matches,
		scene_corners[3] + Point2f((float)img1.cols, 0), scene_corners[0] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);

	return img_matches;
}

void Render()
{
	Player A, B;
	B.hp = 20;

	while (true) {
		Mat img_text;
		char attack[30];
		char defense[30];
		cubemapface->at(3).copyTo(img_text);

		sprintf_s(attack, "A Attack");
		sprintf_s(defense, "B Defense");

		putText(img_text, attack, Point(img_text.cols / 12, img_text.rows / 4), 0, 1, Scalar(0, 0, 0), 2, 8);
		putText(img_text, defense, Point(img_text.cols / 12, img_text.rows / 6), 0, 1, Scalar(0, 0, 0), 2, 8);		

		imageShow("Screen", 300, 300, img_text);
		imwrite("Screen.jpg", img_text);

		waitKey(0);
	}
}

////////////////////////////////////////////////////
// This program demonstrates the usage of SURF_OCL.
// use cpu findHomography interface to calculate the transformation matrix
int main(int argc, char* argv[])
{
	while (turn < gameTurns) {
		UMat img1, img2;
		Mat img;
		Mat srcImage;
		Mat temp;
		Mat tempImg;
		std::string leftName;

		/* 파노라마 이미지 불러오기 */
		if (turn == 0)
		{
			img = imread("Panorama2.jpg"); //자신이 저장시킨 이미지 이름이 입력되어야 함, 확장자까지
			cv::resize(img, srcImage, Size(img.cols * 0.5f, img.rows * 0.5f), 0, 0);
			leftName = "ATrump.jpg";
			temp = imread("Acard.jpg");
		}
		else if (turn == 1)
		{
			img = imread("sph.jpg");
			cv::resize(img, srcImage, Size(img.cols * 1.0f, img.rows * 1.0f), 0, 0);
			leftName = "BTrump.png";
			temp = imread("Bcard.jpg");
		}
		else
		{
			cout << "Background is not available" << endl;
			return 0;
		}


		/* Mat 선언 */
		Mat SphericalToCubemap = CvtSph2Cub(&srcImage);

		std::string outpath = "output.jpg";
		std::string result = "result.jpg";

		imread(leftName, IMREAD_COLOR).copyTo(img1);
		if (img1.empty())
		{
			std::cout << "Couldn't load " << leftName << std::endl;
			return EXIT_FAILURE;
		}

		imwrite("scene.jpg", SphericalToCubemap);

		std::string rightName = "scene.jpg";
		imread(rightName, IMREAD_COLOR).copyTo(img2);
		if (img2.empty())
		{
			std::cout << "Couldn't load " << rightName << std::endl;
			return EXIT_FAILURE;
		}

		double surf_time = 0.;

		//declare input/output
		std::vector<KeyPoint> keypoints1, keypoints2;
		std::vector<DMatch> matches;

		UMat _descriptors1, _descriptors2;
		Mat descriptors1 = _descriptors1.getMat(ACCESS_RW),
			descriptors2 = _descriptors2.getMat(ACCESS_RW);

		//instantiate detectors/matchers
		SURFDetector surf;
		SURFMatcher<BFMatcher> matcher;

		//-- start of timing section

		for (int i = 0; i <= LOOP_NUM; i++)
		{
			if (i == 1) workBegin();
			surf(img1.getMat(ACCESS_READ), Mat(), keypoints1, descriptors1);
			surf(img2.getMat(ACCESS_READ), Mat(), keypoints2, descriptors2);
			matcher.match(descriptors1, descriptors2, matches);
		}
		workEnd();
		std::cout << "FOUND " << keypoints1.size() << " keypoints on first image" << std::endl;
		std::cout << "FOUND " << keypoints2.size() << " keypoints on second image" << std::endl;

		surf_time = getTime();
		std::cout << "SURF run time: " << surf_time / LOOP_NUM << " ms" << std::endl << "\n";


		std::vector<Point2f> corner;
		Mat img_matches = drawGoodMatches(img1.getMat(ACCESS_READ), img2.getMat(ACCESS_READ), keypoints1, keypoints2, matches, corner);
		line(img_matches, inverse_corner[0], inverse_corner[1], Scalar(0, 255, 0), 2, LINE_AA);
		line(img_matches, inverse_corner[1], inverse_corner[2], Scalar(0, 255, 0), 2, LINE_AA);
		line(img_matches, inverse_corner[2], inverse_corner[3], Scalar(0, 255, 0), 2, LINE_AA);
		line(img_matches, inverse_corner[3], inverse_corner[0], Scalar(0, 255, 0), 2, LINE_AA);
		//-- Show detected matches
		imageShow("surf matches", 1200, 800, img_matches);
		imwrite(outpath, img_matches);

		// 일치하는 오브젝트 위치 표시하는 사각형 그리기

		float minX = min(min(corner[0].x, corner[1].x), min(corner[2].x, corner[3].x));
		float minY = min(min(corner[0].y, corner[1].y), min(corner[2].y, corner[3].y));

		float maxX = max(max(corner[0].x, corner[1].x), max(corner[2].x, corner[3].x));
		float maxY = max(max(corner[0].y, corner[1].y), max(corner[2].y, corner[3].y));

		Point2f leftTop = Point2f(minX, minY);
		Point2f rightBottom = Point2f(maxX, maxY);

		/*
			큐브맵 이미지의 패턴 영역 그리기
		*/
		line(img2, corner[0], corner[1], Scalar(0, 0, 255), 2, LINE_AA);
		line(img2, corner[1], corner[2], Scalar(0, 0, 255), 2, LINE_AA);
		line(img2, corner[2], corner[3], Scalar(0, 0, 255), 2, LINE_AA);
		line(img2, corner[3], corner[0], Scalar(0, 0, 255), 2, LINE_AA);

		/*
			큐브맵 이미지 패턴 영역의 바운딩 박스 그리기
		*/
		line(img2, leftTop, Point2f(rightBottom.x, leftTop.y), Scalar(0, 255, 0), 2, LINE_AA);
		line(img2, Point2f(rightBottom.x, leftTop.y), rightBottom, Scalar(0, 255, 0), 2, LINE_AA);
		line(img2, rightBottom, Point2f(leftTop.x, rightBottom.y), Scalar(0, 255, 0), 2, LINE_AA);
		line(img2, Point2f(leftTop.x, rightBottom.y), leftTop, Scalar(0, 255, 0), 2, LINE_AA);
		//imageShow("draw square", 1200, 800, img2);

		int width, height;
		width = rightBottom.x - leftTop.x;
		height = rightBottom.y - leftTop.y;


		cv::resize(temp, tempImg, Size(img1.cols, img1.rows), 0, 0);

		for (int y = leftTop.y; y < rightBottom.y; y++)
		{
			for (int x = leftTop.x; x < rightBottom.x; x++)
			{
				if (y > topEquation(x, corner) && y < bottom(x, corner) && x > left(y, corner) && x < right(y, corner))
				{
					vector<Point2f> xy(1), convert(1);
					xy[0] = Point(x, y);

					perspectiveTransform(xy, convert, inverse);
					SphericalToCubemap.at<Vec3b>(y, x) = tempImg.at<Vec3b>(int(convert[0].y), int(convert[0].x));
				}
			}
		}


		imwrite(result, SphericalToCubemap);
		Mat matched_Cubemap = imread("result.jpg");
		//imageShow("matched cubemap Image", 1200, 800, matched_Cubemap);

		Mat resultSph = CvtCub2Sph(&matched_Cubemap, &srcImage);
		//imageShow("Spherical Image", 1200, 600, resultSph);
		imwrite("sph.jpg", resultSph);
		turn++;
	}

	/* 
		게임 Render
	*/
	
	Render();

	return EXIT_SUCCESS;
}