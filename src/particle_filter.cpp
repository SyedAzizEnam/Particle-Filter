/*
 * particle_filter.cpp
 *
 *  Created on: Jun 8, 2016
 *      Author: Syed Enam
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	num_particles = 1000;
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i< num_particles; i++)
	{
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1;

		particles.push_back(particle);
		weights.push_back(1);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	for (int i = 0; i < num_particles; i++)
	{
		double x_p, y_p, theta_p;

		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		// avoid division by zero
		if (fabs(yaw_rate) > 0.001) {
				x_p = x + velocity/yaw_rate * ( sin (theta + yaw_rate*delta_t) - sin(theta));
				y_p = y + velocity/yaw_rate * ( cos(theta) - cos(theta+yaw_rate*delta_t) );
				theta_p = theta + yaw_rate*delta_t;
		}
		else {
				x_p = x + velocity*delta_t*cos(theta);
				y_p = y + velocity*delta_t*sin(theta);
				theta_p = theta;
		}

		normal_distribution<double> dist_x(x_p, std_pos[0]);
		normal_distribution<double> dist_y(y_p, std_pos[1]);
		normal_distribution<double> dist_theta(theta_p, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.


	for (int i = 0; i < observations.size(); i++)
	{
		double min_distance = 999999;
		for (int j = 0; j < predicted.size(); j++ )
		{
				double distance = dist(predicted[j].x, predicted[j].y, observations[i].x,  observations[i].y);
				if (distance < min_distance) {
					observations[i].id = j;
					min_distance = distance;
				}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	const double std_x = std_landmark[0];
	const double std_y = std_landmark[1];
	const double normalizer = 1.0/sqrt((2* M_PI * std_x * std_y));

	for (int i = 0; i < num_particles; i++)
	{
		//*** Transform observations to map coordinates
		vector<LandmarkObs> transformed_obs;

		for (int j = 0; j < observations.size(); j++)
		{
			int ob_id = observations[j].id;
			double ob_x = observations[j].x;
			double ob_y = observations[j].y;

			double t_x = particles[i].x + ob_x*cos(particles[i].theta) - ob_y*sin(particles[i].theta);
			double t_y = particles[i].y + ob_y*cos(particles[i].theta) + ob_x*sin(particles[i].theta);

			LandmarkObs transformed_ob = {
				ob_id,
				t_x,
				t_y
			};

			transformed_obs.push_back(transformed_ob);
		}

		//*** Find closest landmarks to particle
		vector<Map::single_landmark_s> landmark_list = map_landmarks.landmark_list;
		vector<LandmarkObs> closest_landmarks;

		for (int j = 0; j < landmark_list.size(); j++)
		{
			double distance = dist(particles[i].x, particles[i].y, landmark_list[j].x_f, landmark_list[j].y_f);

			if (distance < sensor_range) {
				LandmarkObs landmark = {
					landmark_list[j].id_i,
					landmark_list[j].x_f,
					landmark_list[j].y_f
				};
				closest_landmarks.push_back(landmark);
			}
		}

		//*** match observations with landmarks
		dataAssociation(closest_landmarks, transformed_obs);

		//*** calculate importance weight

		double weight;

		for (int j = 0; j < transformed_obs.size(); j++)
		{
			int lm_index = transformed_obs[j].id;
			double mx = closest_landmarks[lm_index].x;
			double my = closest_landmarks[lm_index].y;

			double sq_error_x = pow((transformed_obs[j].x - mx), 2);
			double sq_error_y = pow((transformed_obs[j].y - my), 2);

			double gauss_exponent = -(0.5 * sq_error_x/pow(std_x, 2)) - (0.5 * sq_error_y/pow(std_y, 2));

			double weight = normalizer*exp(gauss_exponent);

			particles[i].weight = weight;
			weights[i] = weight;
		}
	}
}

void ParticleFilter::resample() {
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
	discrete_distribution<int> dist(weights.begin(), weights.end());

	vector<Particle> new_particles;
	for (int i = 0; i < num_particles; i++)
	{
			int random_index = dist(gen);
			new_particles.push_back(particles[random_index]);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
