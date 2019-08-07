// This is the Lidar Odometry And Mapping (LOAM) for solid-state lidar (for example: livox lidar),
// which suffer form motion blur due the continously scan pattern and low range of fov.

// Developer: Lin Jiarong  ziv.lin.ljr@gmail.com

//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef LASER_FEATURE_EXTRACTION_H
#define LASER_FEATURE_EXTRACTION_H

#include <cmath>
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <string>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <vector>

#include "livox_feature_extractor.hpp"
#include "tools/common.h"
#include "tools/logger.hpp"

#define LIVOX_NUM 12
#define LIVOX_NUM_MAX 32

using std::atan2;
using std::cos;
using std::sin;
using namespace Common_tools;

class Laser_feature {
public:
    const double m_para_scanPeriod = 0.1;

    int m_if_pub_debug_feature = 1; // TODO: ???

    const int m_para_system_delay = 2;
    int m_para_system_init_count = 0;
    bool m_para_systemInited = false;
    float m_pc_curvature[400000];
    int m_pc_sort_idx[400000];
    int m_pc_neighbor_picked[400000];
    int m_pc_cloud_label[400000];
    int m_if_motion_deblur = 0;
    int m_odom_mode = 0; //0 = for odom, 1 = for mapping
    float m_plane_resolution;
    float m_line_resolution;
    double livox_corners_curvature;
    double livox_surface_curvature;
    double minimum_view_angle;
    File_logger m_file_logger;

    bool m_if_pub_each_line = false;
    int m_lidar_type = 0; // 0 is velodyne, 1 is livox
    int m_laser_scan_number = 64;

    ros::Time m_init_timestamp;

    bool comp(int i, int j) {
        return (m_pc_curvature[i] < m_pc_curvature[j]);
    }

    ros::Publisher m_pub_laser_pc;
    ros::Publisher m_pub_pc_sharp_corner;
    ros::Publisher m_pub_pc_less_sharp_corner;
    ros::Publisher m_pub_pc_surface_flat;
    ros::Publisher m_pub_pc_surface_less_flat;
    ros::Publisher m_pub_pc_removed_pt;
    std::vector<ros::Publisher> m_pub_each_scan;

    ros::Subscriber m_sub_input_laser_cloud;

    double MINIMUM_RANGE = 0.01;

    ros::Publisher m_pub_pc_livox_corners, m_pub_pc_livox_surface, m_pub_pc_livox_full;
    sensor_msgs::PointCloud2 temp_out_msg;
    pcl::VoxelGrid<PointType> m_voxel_filter_for_surface;
    pcl::VoxelGrid<PointType> m_voxel_filter_for_corner;

    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudInAll[LIVOX_NUM_MAX];

    // Calibration config.
    struct CalibrationInfo {
        double x;
        double y;
        double z;
        double rx;
        double ry;
        double rz;
    };
    CalibrationInfo arr_livox_cali_info[LIVOX_NUM_MAX];
    std::vector<int> livox_id_index_vec;

    int init_ros_env() {
        ros::NodeHandle nh;
        m_init_timestamp = ros::Time::now();
        init_livox_lidar_para();

        nh.param<int>("scan_line", m_laser_scan_number, 16);
        nh.param<float>("mapping_plane_resolution", m_plane_resolution, 0.8);
        nh.param<float>("mapping_line_resolution", m_line_resolution, 0.8);
        nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);
        nh.param<int>("if_motion_deblur", m_if_motion_deblur, 1);
        nh.param<int>("odom_mode", m_odom_mode, 0);


        nh.param<double>("corner_curvature", livox_corners_curvature, 0.05);
        nh.param<double>("surface_curvature", livox_surface_curvature, 0.01);
        nh.param<double>("minimum_view_angle", minimum_view_angle, 10);

        printf("scan line number %d \n", m_laser_scan_number);

        if (m_laser_scan_number != 16 && m_laser_scan_number != 64) {
            printf("only support velodyne with 16 or 64 scan line!");
            return 0;
        }

        string log_save_dir_name;
        nh.param<std::string>("log_save_dir", log_save_dir_name, "../");
        m_file_logger.set_log_dir(log_save_dir_name);
        m_file_logger.init("scanRegistration.log");

        m_sub_input_laser_cloud = nh.subscribe<sensor_msgs::PointCloud2>("/laser_points", 10000,
                                                                         &Laser_feature::laserCloudHandler, this);

        m_pub_laser_pc = nh.advertise<sensor_msgs::PointCloud2>("/laser_points_2", 10000);
        m_pub_pc_sharp_corner = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 10000);
        m_pub_pc_less_sharp_corner = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 10000);
        m_pub_pc_surface_flat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 10000);
        m_pub_pc_surface_less_flat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 10000);
        m_pub_pc_removed_pt = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 10000);

        m_pub_pc_livox_corners = nh.advertise<sensor_msgs::PointCloud2>("/pc2_corners", 10000);
        m_pub_pc_livox_surface = nh.advertise<sensor_msgs::PointCloud2>("/pc2_surface", 10000);
        m_pub_pc_livox_full = nh.advertise<sensor_msgs::PointCloud2>("/pc2_full", 10000);

        m_voxel_filter_for_surface.setLeafSize(m_plane_resolution / 2, m_plane_resolution / 2, m_plane_resolution / 2);
        m_voxel_filter_for_corner.setLeafSize(m_line_resolution, m_line_resolution, m_line_resolution);
        if (m_if_pub_each_line) {
            for (int i = 0; i < m_laser_scan_number; i++) {
                ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
                m_pub_each_scan.push_back(tmp);
            }
        }

        return 0;
    }

    ~Laser_feature() {};

    Laser_feature() {
        init_ros_env();
    };

    // 根据最近距离的阈值（eg. 0.01m）,移除距离相隔太近的点。
    template<typename PointT>
    void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                                pcl::PointCloud<PointT> &cloud_out, float thres) {
        if (&cloud_in != &cloud_out) {
            cloud_out.header = cloud_in.header;
            cloud_out.points.resize(cloud_in.points.size());
        }

        size_t j = 0;

        for (size_t i = 0; i < cloud_in.points.size(); ++i) {
            if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y +
                cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
                continue;

            cloud_out.points[j] = cloud_in.points[i];
            j++;
        }

        if (j != cloud_in.points.size()) {
            cloud_out.points.resize(j);
        }

        cloud_out.height = 1;
        cloud_out.width = static_cast<uint32_t>( j );
        cloud_out.is_dense = true;
    }

    void TransfromToLivoxCoordinate(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                                    double tx, double ty, double tz,
                                    double rx, double ry, double rz) {

        for (int i = 0; i < cloud->points.size(); ++i) {
            cloud->points[i].x -= tx;
            cloud->points[i].y -= ty;
            cloud->points[i].z -= tz;

            double tempX = cloud->points[i].x;
            double tempY = cloud->points[i].y;

            cloud->points[i].x = cos(rz) * tempX + sin(rz) * tempY;
            cloud->points[i].y = -sin(rz) * tempX + cos(rz) * tempY;

        }

    }

    void TransfromToHubCoordinate(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                                  double tx, double ty, double tz,
                                  double rx, double ry, double rz) {

        for (int i = 0; i < cloud->points.size(); ++i) {
            double tempX = cloud->points[i].x;
            double tempY = cloud->points[i].y;

            cloud->points[i].x = cos(rz) * tempX - sin(rz) * tempY + tx;
            cloud->points[i].y = sin(rz) * tempX + cos(rz) * tempY + ty;
            cloud->points[i].z += tz;

        }

    }

    // Lidar数据的回调函数，处理单帧lidar数据。
    void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg) {

        double timestamp = laserCloudMsg->header.stamp.toSec();
        // double timestamp = ros::Time::now().toSec();

        // 系统延迟一段时间，再开始处理lidar数据。
        if (!m_para_systemInited) {
            m_para_system_init_count++;

            if (m_para_system_init_count >= m_para_system_delay) {
                m_para_systemInited = true;
            } else
                return;
        }


        for (int i = 0; i < LIVOX_NUM_MAX; ++i) {
            laserCloudInAll[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudIn(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);

        for (int i = 0; i < laserCloudIn->points.size(); ++i) {
            float raw_intensity = laserCloudIn->points[i].intensity;
            int livox_id = round((raw_intensity - float(int(raw_intensity))) * 1000);

            // Transfrom to Livox coordinate.
            laserCloudIn->points[i].z -= arr_livox_cali_info[livox_id].z;

            double tempX = laserCloudIn->points[i].x - arr_livox_cali_info[livox_id].x;
            double tempY = laserCloudIn->points[i].y - arr_livox_cali_info[livox_id].y;

            laserCloudIn->points[i].x =
                    cos(arr_livox_cali_info[livox_id].rz) * tempX + sin(arr_livox_cali_info[livox_id].rz) * tempY;
            laserCloudIn->points[i].y =
                    -sin(arr_livox_cali_info[livox_id].rz) * tempX + cos(arr_livox_cali_info[livox_id].rz) * tempY;

            laserCloudInAll[livox_id]->push_back(laserCloudIn->points[i]);

        }

//        laserCloudInAll[livox_count] = laserCloudInSingle;
//        livox_count++;

//        if (/*livox_count == LIVOX_NUM*/true) {

        pcl::PointCloud<PointType>::Ptr livox_corners_all[LIVOX_NUM];
        pcl::PointCloud<PointType>::Ptr livox_surface_all[LIVOX_NUM];
        pcl::PointCloud<PointType>::Ptr livox_full_all[LIVOX_NUM];

        pcl::PointCloud<PointType>::Ptr livox_corners(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr livox_surface(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr livox_full(new pcl::PointCloud<PointType>());


        // 初始化。
        for (int i = 0; i < LIVOX_NUM; ++i) {
            livox_corners_all[i].reset(new pcl::PointCloud<PointType>());
            livox_surface_all[i].reset(new pcl::PointCloud<PointType>());
            livox_full_all[i].reset(new pcl::PointCloud<PointType>());
        }


        for (int iter = 0; iter < LIVOX_NUM; ++iter) {

            int livox_id = livox_id_index_vec[iter];

//                // 坐标变换到livox坐标系。
//                // Find livox id.
//                float raw_intensity = laserCloudInAll[iter]->points[0].intensity;
//                int livox_id = round((raw_intensity - float(int(raw_intensity))) * 1000);
//
//                TransfromToLivoxCoordinate(laserCloudInAll[iter],
//                                           arr_livox_cali_info[livox_id].x,
//                                           arr_livox_cali_info[livox_id].y,
//                                           arr_livox_cali_info[livox_id].z,
//                                           arr_livox_cali_info[livox_id].rx,
//                                           arr_livox_cali_info[livox_id].ry,
//                                           arr_livox_cali_info[livox_id].rz);


            std::vector<pcl::PointCloud<PointType>> laserCloudScans(m_laser_scan_number);

            Livox_laser m_livox;

            m_livox.thr_corner_curvature = livox_corners_curvature;
            m_livox.thr_surface_curvature = livox_surface_curvature;
            m_livox.minimum_view_angle = minimum_view_angle;


            // 将点云分线到不同的laserCloudScans。livox和传统velodyne的方式有所不同。
            laserCloudScans = m_livox.extract_laser_features(laserCloudInAll[livox_id],
                                                             laserCloudMsg->header.stamp.toSec());


            if (laserCloudScans.empty()) // less than 5 scan
            {
                ROS_WARN_STREAM("laserCloudScansAll.size()= " << laserCloudScans.size() << " !!!");
                continue;
            }

            m_laser_scan_number = laserCloudScans.size() * 1.0;

            //    std::vector< int > scanStartInd( N_SCANS*10, 0 );
            //    std::vector< int > scanEndInd( N_SCANS*10, 0 );
            // 保存玫瑰花瓣scan的起始点的索引坐标。
            std::vector<int> scanStartInd(1000, 0);
            std::vector<int> scanEndInd(1000, 0);

            //        N_SCANS = laserCloudScans.size()/4;
            //N_SCANS = 16;
            scanStartInd.resize(m_laser_scan_number);
            scanEndInd.resize(m_laser_scan_number);
            std::fill(scanStartInd.begin(), scanStartInd.end(), 0);
            std::fill(scanEndInd.begin(), scanEndInd.end(), 0);

            if (m_if_pub_debug_feature) {
                /********************************************
                *    Feature extraction for livox lidar     *
                ********************************************/
                // 将单帧点云分成三份进行处理。
//                int piece_wise = 3;
                // TODO: 为什么分成3份？效果非常垃圾。改成1之后效果明显变好。
                int piece_wise = 1;
                /*if ( m_if_motion_deblur )
                {
                    piece_wise = 1;
                }*/
                vector<float> piece_wise_start(piece_wise); // 小于1的比例量。
                vector<float> piece_wise_end(piece_wise);

                for (int i = 0; i < piece_wise; i++) {
                    int start_scans, end_scans;
                    /*if ( i != 0 )
                    {
                        start_scans = int( ( m_laser_scan_number * (i)  ) / piece_wise ) -1 ;
                        end_scans = int( ( m_laser_scan_number * ( i + 1 ) ) / piece_wise ) - 1;
                    }
                    else
                    {
                        start_scans = 0;
                        end_scans = int( ( m_laser_scan_number * ( 1 ) ) / piece_wise ) ;
                    }*/
                    start_scans = int((m_laser_scan_number * (i)) / piece_wise);
                    end_scans = int((m_laser_scan_number * (i + 1)) / piece_wise) - 1;

                    int start_idx = 0;
                    int end_idx = laserCloudScans[end_scans].size() - 1;
                    piece_wise_start[i] =
                            ((float) m_livox.find_pt_info(
                                    laserCloudScans[start_scans].points[start_idx])->idx) /
                            m_livox.m_pts_info_vec.size();
                    // printf( "Max scan number = %d, start = %d, end  = %d, %d \r\n", m_laser_scan_number, start_scans, end_scans, end_idx );
                    // cout << "Start pt: " << laserCloudScans[ start_scans ].points[ 0 ] << endl;
                    // cout << "End pt: " << laserCloudScans[ end_scans ].points[ end_idx ] << endl;
                    piece_wise_end[i] =
                            ((float) m_livox.find_pt_info(laserCloudScans[end_scans].points[end_idx])->idx) /
                            m_livox.m_pts_info_vec.size();
                }

                for (int i = 0; i < piece_wise; i++) {


                    m_livox.get_features(*livox_corners_all[iter],
                                         *livox_surface_all[iter],
                                         *livox_full_all[iter],
                                         piece_wise_start[i],
                                         piece_wise_end[i]);

                }
            }



            // TODO: 坐标变换回去
            TransfromToHubCoordinate(livox_corners_all[iter],
                                     arr_livox_cali_info[livox_id].x,
                                     arr_livox_cali_info[livox_id].y,
                                     arr_livox_cali_info[livox_id].z,
                                     arr_livox_cali_info[livox_id].rx,
                                     arr_livox_cali_info[livox_id].ry,
                                     arr_livox_cali_info[livox_id].rz);

            TransfromToHubCoordinate(livox_surface_all[iter],
                                     arr_livox_cali_info[livox_id].x,
                                     arr_livox_cali_info[livox_id].y,
                                     arr_livox_cali_info[livox_id].z,
                                     arr_livox_cali_info[livox_id].rx,
                                     arr_livox_cali_info[livox_id].ry,
                                     arr_livox_cali_info[livox_id].rz);

            TransfromToHubCoordinate(livox_full_all[iter],
                                     arr_livox_cali_info[livox_id].x,
                                     arr_livox_cali_info[livox_id].y,
                                     arr_livox_cali_info[livox_id].z,
                                     arr_livox_cali_info[livox_id].rx,
                                     arr_livox_cali_info[livox_id].ry,
                                     arr_livox_cali_info[livox_id].rz);

            // 叠加livox_corners_all[].
            *livox_full += *livox_full_all[iter];
            *livox_surface += *livox_surface_all[iter];
            *livox_corners += *livox_corners_all[iter];

        }


        if (livox_surface->points.size() > 0 &&
            livox_corners->points.size() > 0 &&
            livox_full->points.size() > 10) {

            ros::Time current_time = ros::Time::now();

            pcl::toROSMsg(*livox_full, temp_out_msg);
            temp_out_msg.header.stamp = current_time;
            temp_out_msg.header.frame_id = "/camera_init";
            m_pub_pc_livox_full.publish(temp_out_msg);

            m_voxel_filter_for_surface.setInputCloud(livox_surface);
            m_voxel_filter_for_surface.filter(*livox_surface);
            pcl::toROSMsg(*livox_surface, temp_out_msg);
            temp_out_msg.header.stamp = current_time;
            temp_out_msg.header.frame_id = "/camera_init";
            m_pub_pc_livox_surface.publish(temp_out_msg);

            m_voxel_filter_for_corner.setInputCloud(livox_corners);
            m_voxel_filter_for_corner.filter(*livox_corners);
            pcl::toROSMsg(*livox_corners, temp_out_msg);
            temp_out_msg.header.stamp = current_time;
            temp_out_msg.header.frame_id = "/camera_init";
            m_pub_pc_livox_corners.publish(temp_out_msg);
        }


//        }
    }

    void init_livox_lidar_para() {
        std::string lidar_tpye_name;
        std::cout << "~~~~~ Init livox lidar parameters ~~~~~" << endl;
        if (ros::param::get("lidar_type", lidar_tpye_name)) {
            printf("***** I get lidar_type declaration, lidar_type_name = %s ***** \r\n", lidar_tpye_name.c_str());

            if (lidar_tpye_name.compare("livox") == 0) {
                m_lidar_type = 1;
                std::cout << "Set lidar type = livox" << std::endl;
            } else {
                std::cout << "Set lidar type = velodyne" << std::endl;
                m_lidar_type = 0;
            }
        } else {
            printf("***** No lidar_type declaration ***** \r\n");
            m_lidar_type = 0;
            std::cout << "Set lidar type = velodyne" << std::endl;
        }

//
//            if (ros::param::get("livox_min_dis", m_livox[i].m_livox_min_allow_dis)) {
//                std::cout << "Set livox lidar minimum distance= " << m_livox[i].m_livox_min_allow_dis
//                          << std::endl;
//            }
//
//            if (ros::param::get("livox_min_sigma", m_livox[i].m_livox_min_sigma)) {
//                std::cout << "Set livox lidar minimum sigama =  " << m_livox[i].m_livox_min_sigma
//                          << std::endl;
//            }


        // Livov id index vector.
        livox_id_index_vec.push_back(24); // left front
        livox_id_index_vec.push_back(25);
        livox_id_index_vec.push_back(26);
        livox_id_index_vec.push_back(18); // right front
        livox_id_index_vec.push_back(19);
        livox_id_index_vec.push_back(20);
        livox_id_index_vec.push_back(21); // left back
        livox_id_index_vec.push_back(22);
        livox_id_index_vec.push_back(23);
        livox_id_index_vec.push_back(0);  // right back
        livox_id_index_vec.push_back(1);
        livox_id_index_vec.push_back(2);

        // Calibration init.
        // arr_livox_cali_info[0] = {0, 0,0, 0, 0,0};
//        arr_livox_cali_info[0] = {-0.16, -0.16, 0.737 - 0.0737, 0, 0, -0.75 * M_PI};
//        arr_livox_cali_info[3] = {0.18, -0.18, 0.397 - 0.0397, 0, 0, -0.2417 * M_PI + 0.167 * M_PI};
//        arr_livox_cali_info[4] = {0.18, -0.18, 0.397 - 0.0397, 0, 0, -0.2417 * M_PI};
//        arr_livox_cali_info[5] = {0.18, -0.18, 0.397 - 0.0397, 0, 0, -0.2417 * M_PI - 0.167 * M_PI};
//        arr_livox_cali_info[21] = {-0.16, 0.16, 0.737 - 0.0737, 0, 0, 0.75 * M_PI};
//        arr_livox_cali_info[24] = {0.18, 0.18, 0.397 - 0.0397, 0, 0, 0.25 * M_PI + 0.167 * M_PI};
//        arr_livox_cali_info[25] = {0.18, 0.18, 0.397 - 0.0397, 0, 0, 0.25 * M_PI};
//        arr_livox_cali_info[26] = {0.18, 0.18, 0.397 - 0.0397, 0, 0, 0.25 * M_PI - 0.167 * M_PI};


        arr_livox_cali_info[24] = {0.5, 0.29, 0, 0, 0, (43.0/180.0) * M_PI + 0.167 * M_PI};
        arr_livox_cali_info[25] = {0.5, 0.29, 0, 0, 0, (43.0/180.0) * M_PI};
        arr_livox_cali_info[26] = {0.5, 0.29, 0, 0, 0, (43.0/180.0) * M_PI - 0.167 * M_PI};
        arr_livox_cali_info[18] = {0.5, -0.29, 0, 0, 0, (-45.0/180.0) * M_PI + 0.167 * M_PI};
        arr_livox_cali_info[19] = {0.5, -0.29, 0, 0, 0, (-45.0/180.0) * M_PI};
        arr_livox_cali_info[20] = {0.5, -0.29, 0, 0, 0, (-45.0/180.0) * M_PI - 0.167 * M_PI};
        arr_livox_cali_info[21] = {-0.5, 0.29, 0, 0, 0, (132.0/180.0) * M_PI + 0.167 * M_PI};
        arr_livox_cali_info[22] = {-0.5, 0.29, 0, 0, 0, (132.0/180.0) * M_PI};
        arr_livox_cali_info[23] = {-0.5, 0.29, 0, 0, 0, (132.0/180.0) * M_PI - 0.167 * M_PI};
        arr_livox_cali_info[0] = {-0.5, -0.29, 0, 0, 0, (-136.0/180.0) * M_PI + 0.167 * M_PI};
        arr_livox_cali_info[1] = {-0.5, -0.29, 0, 0, 0, (-136.0/180.0) * M_PI};
        arr_livox_cali_info[2] = {-0.5, -0.29, 0, 0, 0, (-136.0/180.0) * M_PI - 0.167 * M_PI};

        std::cout << "~~~~~ End ~~~~~" << endl;
    }
};

#endif // LASER_FEATURE_EXTRACTION_H
