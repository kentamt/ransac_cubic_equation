/**
 * @brief RANSACアルゴリズムで3次多項式を推定するクラス
 *        外れ点を含む入力データから尤もらしい3次多項式を推定する
 * 
 * @auther kenta
 * @date   2016/12/25
 */

// #ifndef RANSAC3_H
// #define RANSAC3_H


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <time.h>

#include <opencv2/core/core.hpp> // 連立方程式ソルバ用
#include <opencv2/highgui/highgui.hpp>


using namespace std;

 
/**
 * @brief RANSACで直線を推定するクラス
 */
class RANSAC3{

public:

  RANSAC3(){init();}
  ~RANSAC3(){}
  
  /**
   * @brief 初期化
   */
  void init(){
    num_inliers_th_ = 300;
    threshold_= 8000000;
    loop_num_       = 500;	// ループ回数
    error_th_	    = 10.0;	// inlierの閾値
  }

  double get_a(){return a_;}
  double get_b(){return b_;}
  double get_c(){return c_;}
  double get_d(){return d_;}
  vector<cv::Point> inliers_; //（簡易なので）直接読めるようにしておくTODO: cv::Pointはintなので、floatにも対応すること
  vector<cv::Point> outliers_; //（簡易なので）直接読めるようにしておく
  
  /**
   * @brief RANSACを実行する関数
   * @param[out] a
   * @param[out] b
   */ 
  void do_ransac(){
    solve_equations(a_, b_, c_, d_);
  }

  /**
   * @brief RANSACをするための入力データを準備する関数
   */
  void set_data( vector<cv::Point> input_data){
    data_ = input_data;
  }

  void make_simulation_data(){
 
    // make noise 
    srand((int)time(NULL));
    double temp;
    cv::Point p;
    for (int i = 0; i < 300; i++){ 
      p.x = get_random(1, 300);
      p.y = get_random(1, 500);
      data_.push_back(p);
    }
 
    // make true data 
    // a x^3 + b x^2 + c x ^1 + d x^0
    double k = 0.0005;
    aa_ = 1*k;
    bb_ = -300*k;
    cc_ = 20000*k;
    dd_ = 240;

    
    for (int x = 0; x < 400; x++){

      // add noise
      temp = (double)get_random(0, 10);
      if (x % 2 == 0)
      	temp = (-1) * temp;

      // create true data
      p.x = x;
      p.y = aa_*x*x*x + bb_*x*x + cc_*x + dd_ + temp;
      data_.push_back(p);
    }  
  }


  /**
   * @brief 結果をgnuplotで可視化する関数. killall -9 gnuplotで終わること。
   * @param
   */
  void draw_graph(){
 
    // file pointer for stdout 
    FILE *gp;
    int data_num = data_.size();
    int inliers_num = inliers_.size();
 
    // settings
    gp = popen("gnuplot -persist", "w");

    // set output image
    fprintf(gp, "set terminal png\n");
    fprintf(gp, "set out \"%s_ransac_result.png\"\n", get_time_string().c_str());

    // fprintf(gp, "set size square\n");
    fprintf(gp, "set title \"RANSAC\"\n");

    // set range
    int x_max = 255;
    int y_max = 480;
    fprintf(gp, "set xrange[0:%d]\n", x_max);
    fprintf(gp, "set yrange[0:%d]\n", y_max);

    // plot input data
    for (int i = 0; i < data_num; i++){
      fprintf(gp, "set label %d point pt 7 lc rgb \"gray\" at %f,%f\n",  i + 1, (float)data_[i].x, (float)data_[i].y);
    }

    // plot inliers
    for (int i = 0; i < inliers_num; i++){
      fprintf(gp, "set label %d point pt 7 lc rgb \"spring-green\" at %f,%f\n",  i + data_num, (float)inliers_[i].x, (float)inliers_[i].y);
    }

    // plot estimated line
    fprintf(gp, "plot %f * x * x * x + %f * x * x + %f * x + %f lw 2, %f * x * x * x + %f * x * x + %f * x + %f lw 2 \n", 
    	    a_, b_, c_, d_, // 推定結果
    	    aa_, bb_, cc_, dd_);    // 真値

  }
 

private:
  
  vector<cv::Point>	data_;	// input data_
  int			num_inliers_th_;
  double		threshold_;
  double		a_, b_, c_, d_; // estimated value
  double		aa_, bb_, cc_, dd_; // true value
  int			loop_num_;
  double		error_th_;
  double		max_error_th_;

  /**
   * @brief ある範囲からランダムに値を取得する関数
   * @param
   */
  int get_random(int min, int max){
    return min + (int)(rand()*(max - min + 1.0) / (1.0 + RAND_MAX));
  }



  void solve_equations(double &a, double &b, double &c, double &d){
    
    cv::Mat left_side, right_side, solution; // 連立方程式の左辺、右辺、解
    int max_inliers_num = 0; // inlier    
    double best_a; 
    double best_b;
    double best_c;
    double best_d;
    vector<cv::Point> best_inliers; 
   
    for (int i = 0; i < loop_num_; i++){

      int num_data = data_.size();

      // 3点をランダムに選ぶ（ランダムに2次多項式を作る）
      int x1 = get_random(0, num_data-1);
      int x2 = get_random(0, num_data-1);
      int x3 = get_random(0, num_data-1);
      int x4 = get_random(0, num_data-1);
 
      // 連立方程式の左右をそれぞれ行列形式で表現
      left_side  = (cv::Mat_<double>(4, 4)
		    << 
		    data_[x1].x * data_[x1].x * data_[x1].x  , data_[x1].x * data_[x1].x, data_[x1].x, 1, 
		    data_[x2].x * data_[x2].x * data_[x2].x  , data_[x2].x * data_[x2].x, data_[x2].x, 1, 
		    data_[x3].x * data_[x3].x * data_[x3].x  , data_[x3].x * data_[x3].x, data_[x3].x, 1, 
		    data_[x4].x * data_[x4].x * data_[x4].x  , data_[x4].x * data_[x4].x, data_[x4].x, 1);

      right_side = (cv::Mat_<double>(4, 1)
		    << 
		    data_[x1].y, 
		    data_[x2].y, 
		    data_[x3].y,
		    data_[x4].y);
 
      // 連立方程式を解く
      solution = cv::Mat(4, 1, CV_64FC1);
      cv::solve(left_side, right_side, solution);
      a = solution.at<double>(0, 0);
      b = solution.at<double>(1, 0);
      c = solution.at<double>(2, 0);
      d = solution.at<double>(3, 0);


      for (int j = 0; j < num_data; j++){
 
	// ある範囲内であればinlierとする TODO: y方向誤差のみを評価している 点と直線の距離にすべき
	// if (data_[j].y <= (a * data_[j].x + b + error_th_) &&  data_[j].y >= (a * data_[j].x + b - error_th_) ){
	double value = a * data_[j].x * data_[j].x * data_[j].x  + b * data_[j].x * data_[j].x + c * data_[j].x + d;
	if (data_[j].y <= ( value + error_th_) &&  data_[j].y >= (value - error_th_) ){

	  push_data(inliers_, data_[j].x, data_[j].y);

	}
      }


      // inliers_numがmax_inlers_numより大きければ、値を更新
      int num_inliers = inliers_.size();      
      if ( num_inliers > max_inliers_num ){       
	max_inliers_num = num_inliers;
	best_a = a;
	best_b = b;
	best_c = c;
	best_d = d;
	best_inliers = inliers_;
      }      
      // 1試行ごとにインライアを破棄する
      inliers_.clear();
    }
    
    // 最も良かったinliersを使って再度推定する
    inliers_ = best_inliers;
    a = best_a;
    b = best_b;
    c = best_c;
    d = best_d;

    // re_estimate(a, b);

  }


  /**
   * @brief inlierを使って、最小二乗法で直線を再計算する関数
   * @param
   */  
  void re_estimate(double &a, double &b, double &c, double &d){
  }


  /**
   * @brief dataをpushする関数
   * @param
   */  
  void push_data(vector<cv::Point>& data, double x, double y){
    cv::Point p;
    p.x = x;
    p.y = y;
    data.push_back(p);
  }


  /**
   * @brief 現在時刻をstring型で取得する関数
   */
  string get_time_string(){
    //時刻取得用
    string time_str;
    char buff[128];

    //現在時刻取得
    time_t now = time(NULL);
    struct tm *pnow = localtime(&now);
    sprintf(buff, "%04d%02d%02d%02d%02d%02d",
	    pnow->tm_year + 1900, 
	    pnow->tm_mon + 1, 
	    pnow->tm_mday,
	    pnow->tm_hour, 
	    pnow->tm_min, 
	    pnow->tm_sec);
    
    time_str = buff;
    return time_str;
  }
  
  
 
};

  // test  ===

  int main(int argc, char **argv){

    // 初期化    
    RANSAC3 ransac;
    ransac.make_simulation_data();

    // RANSAC
    ransac.do_ransac();
  
    // 結果
    ransac.draw_graph();
    double a = ransac.get_a();
    double b = ransac.get_b();
    double c = ransac.get_c();
    double d = ransac.get_d();
    cout << "num of inliers: " << ransac.inliers_.size() << endl;
    cout << "a: " << a << endl; 
    cout << "b: " << b << endl;
    cout << "c: " << c << endl;
    cout << "d: " << d << endl;

    return 0;
  }

  // #endif // RANSAC3
