//
// Created by Lazurite on 4/29/2022.
//

#include "basetrack.h"
#include "map"
#include "vector"
#include "deque"

typedef enum TrackState {
    New,
    Tracked,
    Lost,
    Removed
} TrackState;

class BaseTrack {
public:
    int _count = 0;

    int track_id = 0;
    bool is_activated = false;
    TrackState state = New;

    int start_frame = 0;
    int frame_id = 0;


    int end_frame() {
        return frame_id;
    }

    int next_id() {
        return _count++;
    }

    void mark_lost() {
        state = Lost;
    }

    void mark_removed() {
        state = Removed;
    }

    virtual void activate() = 0;

    virtual void predict() = 0;

    virtual void update() = 0;

};


class STrack : public BaseTrack {
public:
    KalmanFilter shared_kalman;
    KalmanFilter *kalman_filter = nullptr;
    Eigen::Vector4f _tlwh;
    double score;
    int n_history = 0;
    int buffer_size = 0;
    std::deque<Eigen::VectorXf> history;
    Eigen::VectorXf curr_feat;
    Eigen::VectorXf smooth_feat;
    double alpha = 0.9;
    Eigen::Matrix<double, 8, 1> mean;
    Eigen::Matrix<double, 8, 1> mean_state;
    Eigen::Matrix<double, 8, 8, Eigen::RowMajor> covariance;

    void update_features(Eigen::Ref<Eigen::VectorXf> feat);

    STrack(Eigen::Ref<Eigen::Vector4f> tlwh, double score, Eigen::Ref<Eigen::VectorXf> temp_feat, int buffer_size);

    void feature_append(Eigen::VectorXf &feat);

    Eigen::Vector4d tlwh_to_xyah(Eigen::Vector4f &tlwh){
        Eigen::Vector4d xyah;
        xyah << tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2, tlwh[2]/tlwh[3], tlwh[3];
        return xyah;
    }

    void foo(const Eigen::Ref<Eigen::Vector4d> &measurement){
        ;
    }

    void activate(KalmanFilter *kalman_filter_, int frame_id){
        kalman_filter = kalman_filter_;
        track_id = next_id();

        Eigen::Matrix<double, 8, 1> mean;
        Eigen::Matrix<double, 8, 8, Eigen::RowMajor> covariance;
        Eigen::Vector4d xyah = tlwh_to_xyah(_tlwh);
        foo(xyah);
    }

    void predict() {
        mean_state = mean;
        if (state != Tracked){

        }
    }
};

STrack::STrack(Eigen::Ref<Eigen::Vector4f> tlwh, double score, Eigen::Ref<Eigen::VectorXf> temp_feat, int buffer_size)
        : score(score), buffer_size(buffer_size) {
    _tlwh = tlwh;
    // First init smooth_feat to avoid judging whether it is empty
    smooth_feat = temp_feat.normalized();
    update_features(temp_feat);
}

void STrack::update_features(Eigen::Ref<Eigen::VectorXf> feat) {
    /*
     * Update features
     * Update curr_feat, smooth_feat, and add smooth_feat to history
     */
    // Normalize feature
    curr_feat = feat.normalized();
    feature_append(curr_feat);
    smooth_feat = alpha * smooth_feat + (1 - alpha) * curr_feat;
    smooth_feat.normalize();

}

void STrack::feature_append(Eigen::VectorXf &feat) {
    /*
     * Add feature to history
     * If history is full, remove the oldest one
     */
    history.push_back(feat);
    n_history++;
    if (n_history > buffer_size) {
        history.pop_front();
        n_history--;
    }
}


void init_module_basetrack(py::module &m) {
    m.def("is_basetrack_init", []() {
        return true;
    });

    py::class_<STrack>(m, "STrack");
}