/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <map>

#include "helper_functions.h"

#define _DEBUG_SWITCH false

using std::string;
using std::vector;

using std::normal_distribution;
using std::default_random_engine;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine gen;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  num_particles = 1000;  // TODO: Set the number of particles

  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen); 
    p.weight = 1;
    
    particles.push_back(p);
    weights.push_back(p.weight);

    is_initialized = true;

    // Print your samples to the terminal.
    if(_DEBUG_SWITCH) {
      std::cout << "Init #" << i << ": " << p.x << " " << p.y << " " 
                << p.theta << std::endl;
    }
  }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // Note: I guess the noise should be v_noise and yawrate_noise
  // And the calculate of pred values shoule be based on noisy v and yawrate
  // However it is defined as 'std_pos'. So I implement the function like this

  std::default_random_engine gen;
  //normal_distribution<double> dist_v(velocity, 0.3);
  //normal_distribution<double> dist_yawrate(yaw_rate, 0.01);

  double pred_x, pred_y, pred_theta;
  //double noise_velocity, noise_yawrate;

  if(_DEBUG_SWITCH) {
    std::cout << "prediction: delta_t:" << delta_t << "\t V:" << velocity << "\t yaw_rate:" << yaw_rate << std::endl;
  }

  for (int i = 0; i < num_particles; ++i) {
    
    Particle p = particles[i];
    
    //noise_velocity = dist_v(gen);
    //noise_yawrate = dist_yawrate(gen);
    if(yaw_rate != 0.0) {
      pred_x = p.x + (velocity/yaw_rate) * (sin(p.theta + yaw_rate * delta_t)-sin(p.theta));
      pred_y = p.y + (velocity/yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
      pred_theta = p.theta + yaw_rate * delta_t;
    }else{
      pred_x = p.x + velocity * delta_t * cos(p.theta);
      pred_y = p.y + velocity * delta_t * sin(p.theta);
      pred_theta = p.theta;
    }

    normal_distribution<double> dist_x(pred_x, std_pos[0]);
    normal_distribution<double> dist_y(pred_y, std_pos[1]);
    normal_distribution<double> dist_theta(pred_theta, std_pos[2]);



    if(_DEBUG_SWITCH) {
      std::cout << "Before #" << p.id << ": " << p.x << " " << p.y << " " \
                << p.theta << std::endl;
      std::cout << "After #" << p.id << ": " << pred_x << " " << pred_y << " " \
                << pred_theta << std::endl;       
    }
    
    //p.x = pred_x;
    //p.y = pred_y;
    //p.theta = pred_theta; 
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen); 

    particles[i] = p;

  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  int pred_size = predicted.size();
  int obs_size = observations.size();

  for(int i=0; i<obs_size; i++) {
    observations[i].id = -1;
    double min_dist = -1.0;
    //std::cout << "obs:(" << observations[i].x << "," << observations[i].y << ")" << std::endl;
    for(int j=0; j<pred_size; j++) {
      double dist_ = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
      //std::cout << "pred landmark:(" << predicted[j].x << "," << predicted[j].y << ")\t dist:" << dist_ << std::endl;
      if(min_dist < 0) {
        observations[i].id = predicted[j].id;
        min_dist = dist_;
      } else {
        if(dist_ < min_dist) {
          observations[i].id = predicted[j].id;
          min_dist = dist_;
          //std::cout<< "nearest:" << std::endl;
        }
      }
    }
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  if(_DEBUG_SWITCH) {
      std::cout << "Start update" << std::endl;
  }
  int map_size = map_landmarks.landmark_list.size();
  int obs_size = observations.size();

  for(int i = 0; i < num_particles; i++) {
    Particle p = particles[i];

    // Add the landmarks within the sensor range
    double sensor_range_x_max = p.x + sensor_range;
    double sensor_range_x_min = p.x - sensor_range;
    double sensor_range_y_max = p.y + sensor_range;
    double sensor_range_y_min = p.y - sensor_range;
    
    vector<LandmarkObs> pred_landmarks;
    for(int j = 0; j < map_size; j++) {
      if(map_landmarks.landmark_list[j].x_f > sensor_range_x_min && \
         map_landmarks.landmark_list[j].x_f < sensor_range_x_max && \
         map_landmarks.landmark_list[j].y_f > sensor_range_y_min && \
         map_landmarks.landmark_list[j].y_f < sensor_range_y_max) {
           LandmarkObs pred_lm;
        
           pred_lm.id = map_landmarks.landmark_list[j].id_i;
        //std::cout<< "landmark index:" << j << "\t id:" << map_landmarks.landmark_list[j].id_i << std::endl;
           pred_lm.x = map_landmarks.landmark_list[j].x_f;
           pred_lm.y = map_landmarks.landmark_list[j].y_f;
           pred_landmarks.push_back(pred_lm);
      }
    }

    if(_DEBUG_SWITCH) {
      std::cout << "Particle #" << p.id << ":(" << p.x << "," << p.y <<")";
      std::cout << "Number of landmarks within sensor range:" << pred_landmarks.size() << std::endl;
    }

    // Transform the observation to map coordinates
    vector<LandmarkObs> obs;
    for(int j = 0; j < obs_size; j++) {
      obs.push_back(HomoTrans(p.x, p.y, p.theta, observations[j]));
    }

    dataAssociation(pred_landmarks, obs);

    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    double pweight = 1;
    for(int j = 0; j < obs_size; j++) {
      associations.push_back(obs[j].id);
      sense_x.push_back(obs[j].x);
      sense_y.push_back(obs[j].y);
      double lm_x = map_landmarks.landmark_list[obs[j].id - 1].x_f;
      double lm_y = map_landmarks.landmark_list[obs[j].id - 1].y_f;
      double prob = multiv_prob(std_landmark[0], std_landmark[1],lm_x, lm_y, obs[j].x, obs[j].y);
      pweight = pweight * prob;
      if(_DEBUG_SWITCH) {
        std::cout << "landmark pos: (" << lm_x << "," << lm_y << ")\t obs pos: (" << obs[j].x << "," << obs[j].y << ")\t multiv_prob: " << prob << std::endl;
      }
    }
    SetAssociations(p, associations, sense_x, sense_y);
    p.weight = pweight;
    weights[i] = pweight;

    if(_DEBUG_SWITCH) {
        std::cout << "Final weight for #" << i << "_th:" << p.weight << std::endl;
    }

    particles[i] = p;
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::random_device rd;
  std::mt19937 gen(rd());
  


  std::vector<Particle> new_Particles;
  if(_DEBUG_SWITCH) {
    std::cout<< "Start resample" << std::endl;
    for(int i=0; i<num_particles; i++) {
      std::cout<< i <<"_th particle weight:" << weights[i] << std::endl;
    }
  }

  std::discrete_distribution<> d(weights.begin(), weights.end());
  std::map<int, int> m;

  for(int n=0; n<num_particles; ++n) {
    int new_particle_index = d(gen);
    Particle sampled_p = particles[new_particle_index];
    sampled_p.id = n;
    ++m[new_particle_index];
    new_Particles.push_back(sampled_p);
  }
  
  if(_DEBUG_SWITCH) {
    for(auto p : m) {
        std::cout << p.first << " generated " << p.second << " times\n";
    }
  }

  particles = new_Particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}