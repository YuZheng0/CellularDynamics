#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <random>

//*************Set Parameters**************
const double LENGTH = 1;    // The length of box
const double DENSITY = 0.4; // The density of system
const double RADIUS = 0.1;  // The radius of particles

const double DELTATIME = 1; // particles[i].position[0] += DELTATIME * particles[i].velocity[0];
const int STEP = 1000;      // The number of movement steps
const double PI = 3.1415926;
const double AREAOFBOX = LENGTH * LENGTH;
const int NUMBER = int(AREAOFBOX * DENSITY / (PI * RADIUS * RADIUS));
const double V = 0.01;
//*************Set Parameters**************

//********Set Random Numbers*********
std::random_device rd;
std::default_random_engine gen(rd());
std::normal_distribution<double> norm_distribution_angle(0, PI / 64);
std::uniform_real_distribution<float> uniform_distribution(0.0, 1.0);
//********Set Random Numbers*********

struct Particle
{
    double position[2];
    double velocity[2];
    double force[2];
};

double boundaryDisplacementX(Particle a, Particle b)
{
    double deltax = a.position[0] - b.position[0];

    if (std::abs(deltax) > LENGTH / 2)
    {
        if (deltax > 0)
            deltax -= LENGTH;
        else
            deltax += LENGTH;
    }
    return deltax;
}

double boundaryDisplacementY(Particle a, Particle b)
{
    double deltay = a.position[1] - b.position[1];

    if (std::abs(deltay) > LENGTH / 2)
    {
        if (deltay > 0)
            deltay -= LENGTH;
        else
            deltay += LENGTH;
    }
    return deltay;
}

double boundaryDistance(Particle a, Particle b)
{
    double deltax = boundaryDisplacementX(a, b);
    double deltay = boundaryDisplacementY(a, b);

    return std::sqrt(deltax * deltax + deltay * deltay);
}

/*double distance(particle a, particle b)
{
    return std::sqrt((a.position[0] - b.position[0]) * (a.position[0] - b.position[0]) + (a.position[1] - b.position[1]) * (a.position[1] - b.position[1]));
}*/

bool isOverlap(Particle a, Particle b)
{
    return (boundaryDistance(a, b) < 2 * RADIUS);
}

void initialization(Particle particles[NUMBER])
{
    int i = 0;
    while (i < NUMBER)
    {
        particles[i].position[0] = LENGTH * (uniform_distribution(gen));
        particles[i].position[1] = LENGTH * (uniform_distribution(gen));

        bool hasOverlap = false;
        for (int j = 0; j < i; j++)
        {
            if (i != j && isOverlap(particles[i], particles[j]))
            {
                hasOverlap = true;
                break;
            }
        }
        if (!hasOverlap)
        {
            i++;
            //std::cout << "Add: " << i << " out of " << NUMBER << std::endl;
        }
    }

    for (int ii = 0; ii < NUMBER; ii++)
    {
        particles[ii].velocity[0] = V * LENGTH * (uniform_distribution(gen) - 0.5);
        particles[ii].velocity[1] = V * LENGTH * (uniform_distribution(gen) - 0.5);
        particles[ii].force[0] = 0;
        particles[ii].force[1] = 0;
    }
}

void correctOverlap(Particle &a, Particle &b)
{
    double d = boundaryDistance(a, b);
    double deltax = boundaryDisplacementX(a, b);
    double deltay = boundaryDisplacementY(a, b);
    double deltaD = 2 * RADIUS - d;

    a.position[0] += 0.5 * deltaD * deltax / d;
    a.position[1] += 0.5 * deltaD * deltay / d;
    b.position[0] += 0.5 * deltaD * (-deltax) / d;
    b.position[1] += 0.5 * deltaD * (-deltay) / d;
}

double getAngle(double x, double y)
{
    double angle = atan(y / x);
    if (x < 0)
        angle += PI;

    return angle;
}

void move(Particle particles[NUMBER])
{
    int i, j;
#pragma omp parallel for private(i)
    for (i = 0; i < NUMBER; i++)
    {
        //Update positions
        particles[i].position[0] += DELTATIME * particles[i].velocity[0];
        particles[i].position[1] += DELTATIME * particles[i].velocity[1];

        for (j = 0; j < NUMBER; j++)
        {
            if (j != i)
            {
                if (isOverlap(particles[i], particles[j]))
                {
                    correctOverlap(particles[i], particles[j]);
                }
            }
        }

        //rotation diffusion
        double velocityAngle;
        double v = std::sqrt(particles[i].velocity[0] * particles[i].velocity[0] + particles[i].velocity[1] * particles[i].velocity[1]);
        velocityAngle = getAngle(particles[i].velocity[0], particles[i].velocity[1]);
        velocityAngle += norm_distribution_angle(gen);
        particles[i].velocity[0] = v * cos(velocityAngle);
        particles[i].velocity[1] = v * sin(velocityAngle);
        ////////////////////

        if (particles[i].position[0] >= LENGTH)
            particles[i].position[0] = particles[i].position[0] - LENGTH;
        if (particles[i].position[0] < 0)
            particles[i].position[0] = particles[i].position[0] + LENGTH;
        if (particles[i].position[1] >= LENGTH)
            particles[i].position[1] = particles[i].position[1] - LENGTH;
        if (particles[i].position[1] < 0)
            particles[i].position[1] = particles[i].position[1] + LENGTH;
    }
}

int main()
{
    // double startTime, endTime;
    //startTime = omp_get_wtime();
    const int numSims = 10000;
    Particle particles[NUMBER];
    for (int sim = 1; sim < numSims; sim++)
    {
        std::cout << "Running simulation " << sim << std::endl;

        initialization(particles);

        std::ofstream fout1("data" + std::to_string(sim) + "/position0.txt");

        for (int i = 0; i < NUMBER; i++)
        {
            fout1 << particles[i].position[0] << " " << particles[i].position[1] << " " << particles[i].velocity[0] << " " << particles[i].velocity[1] << std::endl;
        }
        fout1.close();

        for (int step = 0; step < STEP; step++)
        {
            for (int i = 0; i < NUMBER; i++)
            {
                particles[i].force[0] = 0;
                particles[i].force[1] = 0;
            }
            move(particles);

            const int tick = 10;

            if ((step + 1) % tick == 0)
            {
                // std::cout << "Step " << step + 1 << " out of " << STEP << std::endl;
                const std::string fileName = "data" + std::to_string(sim) + "/position" + std::to_string(step + 1) + ".txt";
                std::ofstream fout(fileName);
                for (int i = 0; i < NUMBER; i++)
                {
                    fout << particles[i].position[0] << " " << particles[i].position[1] << " " << particles[i].velocity[0] << " " << particles[i].velocity[1] << std::endl;
                }
                fout.close();
            }
        }
    }

    // endTime = omp_get_wtime();
    // std::cout << "Total time: " << endTime - startTime << " s" << std::endl;
}
