#include <iostream>
#include <fstream>
#include <cmath>

//*************Set Parameters**************
const double LENGTH = 1;                    // The length of box
const double DENSITY = 0.25;                // The density of system
const double RADIUS = 0.05;                 // The radius of particles
const double DIFFCOEFF = 1;                 // The diffusion coefficient
const double KCON = RADIUS * 0.00001;       // The coefficient for contact force
const double KECM = RADIUS * 0.00001;       // The coefficient for ECM force
const double DIFFCOEFF2 = RADIUS * 0.00001; // The diffusion coefficient for random force
const double EPS = 0.002;                   // A samll number for preventing division divergence
const double FSELF = RADIUS * 0.00001;      // The self propulsion force
const double ANGLE = 1;                     // For ECM force
const double MOVELENGTH = 0.01;             // particles[i].position[0] += MOVELENGTH * particles[i].velocity[0];
const int STEP = 10000;                     // The number of movement steps
const double PI = 3.1415926;
const int NUMBER = int(DENSITY / (PI * RADIUS * RADIUS));
//*************Set Parameters**************

struct Particle
{
    double position[2];
    double velocity[2];
    double force[2];
};

double boundaryDistance(Particle a, Particle b)
{
    double deltax, deltay;
    deltax = std::abs(a.position[0] - b.position[0]);
    deltay = std::abs(a.position[1] - b.position[1]);
    if (deltax >= LENGTH / 2)
        deltax = LENGTH - deltax;
    if (deltay >= LENGTH / 2)
        deltay = LENGTH - deltay;
    return std::sqrt(deltax * deltax + deltay * deltay);
}

/*double distance(particle a, particle b)
{
    return std::sqrt((a.position[0] - b.position[0]) * (a.position[0] - b.position[0]) + (a.position[1] - b.position[1]) * (a.position[1] - b.position[1]));
}*/

bool isOverlap(Particle a, Particle b)
{
    if (boundaryDistance(a, b) < 2 * RADIUS)
        return true;
    else
        return false;
}

bool isOppsiteMove(Particle a, Particle b)
{
    double d = boundaryDistance(a, b);
    double dv = std::sqrt(a.velocity[0] * a.velocity[0] + a.velocity[1] * a.velocity[1]);
    double deltax, deltay, deltax1, deltay1;
    double cosTheta;
    deltax = a.position[0] - b.position[0];
    deltay = a.position[1] - b.position[1];
    if (std::abs(deltax) >= LENGTH / 2)
        deltax1 = LENGTH - deltax;
    if (std::abs(deltay) >= LENGTH / 2)
        deltay1 = LENGTH - deltay;
    if (deltax >= 0)
        deltax = -deltax1;
    else
        deltax = deltax1;
    if (deltay >= 0)
        deltay = -deltay1;
    else
        deltay = deltay1;
    cosTheta = (-deltax * a.velocity[0] + (-deltay * a.velocity[1])) / (d * dv);
    if (cosTheta >= cos(ANGLE * PI / 180) and (a.velocity[0] * b.velocity[0] + a.velocity[1] * b.velocity[1] < 0))
        return true;
    else
        return false;
}

void initialization(Particle particles[NUMBER])
{

    int i = 0;
    while (i < NUMBER)
    {
        particles[i].position[0] = LENGTH * (std::rand() / double(RAND_MAX));
        particles[i].position[1] = LENGTH * (std::rand() / double(RAND_MAX));

        bool hasOverlap = false;
        for (int j = 0; j < i; j++)
        {
            if (isOverlap(particles[i], particles[j]))
            {
                hasOverlap = true;
                break;
            }
        }
        if (!hasOverlap)
        {
            i++;
            std::cout << "Add: " << i << " out of " << NUMBER << std::endl;
        }

        for (int ii = 0; ii < NUMBER; ii++)
        {
            particles[ii].velocity[0] = LENGTH * ((std::rand() / double(RAND_MAX)) - 0.5);
            particles[ii].velocity[1] = LENGTH * ((std::rand() / double(RAND_MAX)) - 0.5);
            particles[ii].force[0] = 0;
            particles[ii].force[1] = 0;
        }
    }

    std::ofstream fout2("oriPosition.txt");
    for (int i = 0; i < NUMBER; i++)
    {
        fout2 << particles[i].position[0] << " " << particles[i].position[1] << std::endl;
    }
    fout2.close();
}

void forceContactECM(Particle particles[NUMBER])
{
    for (int i = 0; i < NUMBER; i++)
    {
        for (int j = 0; j < NUMBER; j++)
        {
            if (isOverlap(particles[i], particles[j]))
            {
                double d = boundaryDistance(particles[i], particles[j]);
                double deltax, deltay, deltax1, deltay1;
                deltax = particles[i].position[0] - particles[j].position[0];
                deltay = particles[i].position[1] - particles[j].position[1];
                if (std::abs(deltax) >= LENGTH / 2)
                {
                    deltax1 = LENGTH - deltax;
                    if (deltax >= 0)
                        deltax = -deltax1;
                    else
                        deltax = deltax1;
                }
                if (std::abs(deltay) >= LENGTH / 2)
                {
                    deltay1 = LENGTH - deltay;
                    if (deltay >= 0)
                        deltay = -deltay1;
                    else
                        deltay = deltay1;
                }
                particles[i].force[0] += DIFFCOEFF * KCON * std::abs(2 * RADIUS - d) * deltax / d;
                particles[i].force[1] += DIFFCOEFF * KCON * std::abs(2 * RADIUS - d) * deltay / d;
            }
            else
            {
                if (isOppsiteMove(particles[i], particles[j]))
                {
                    double d = boundaryDistance(particles[i], particles[j]);
                    double deltax, deltay, deltax1, deltay1;
                    deltax = particles[i].position[0] - particles[j].position[0];
                    deltay = particles[i].position[1] - particles[j].position[1];
                    if (std::abs(deltax) >= LENGTH / 2)
                    {
                        deltax1 = LENGTH - deltax;
                        if (deltax >= 0)
                            deltax = -deltax1;
                        else
                            deltax = deltax1;
                    }
                    if (std::abs(deltay) >= LENGTH / 2)
                    {
                        deltay1 = LENGTH - deltay;
                        if (deltay >= 0)
                            deltay = -deltay1;
                        else
                            deltay = deltay1;
                    }
                    particles[i].force[0] += DIFFCOEFF * (KECM / d) * (-deltax / d);
                    particles[i].force[1] += DIFFCOEFF * (KECM / d) * (-deltay / d);
                }
            }
        }
    }
}

void forceSelf(Particle particles[NUMBER])
{
    for (int i = 0; i < NUMBER; i++)
    {
        double v = sqrt(particles[i].velocity[0] * particles[i].velocity[0] + particles[i].velocity[1] * particles[i].velocity[1]);
        particles[i].force[0] += DIFFCOEFF * FSELF * particles[i].velocity[0] / v;
        particles[i].force[1] += DIFFCOEFF * FSELF * particles[i].velocity[1] / v;
    }
}

void forceRandom(Particle particles[NUMBER])
{
    for (int i = 0; i < NUMBER; i++)
    {
        particles[i].force[0] += DIFFCOEFF2 * LENGTH * (std::rand() / double(RAND_MAX)) - 0.5;
        particles[i].force[1] += DIFFCOEFF2 * LENGTH * (std::rand() / double(RAND_MAX)) - 0.5;
    }
}

void move(Particle particles[NUMBER])
{
    for (int i = 0; i < NUMBER; i++)
    {
        particles[i].velocity[0] = particles[i].force[0];
        particles[i].velocity[1] = particles[i].force[1];
        //Update positions
        particles[i].position[0] += MOVELENGTH * particles[i].velocity[0];
        particles[i].position[1] += MOVELENGTH * particles[i].velocity[1];
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
    srand((unsigned)time(NULL));
    Particle particles[NUMBER];
    initialization(particles);
    for (int step = 0; step < STEP; step++)
    {
        for (int i = 0; i < NUMBER; i++)
        {
            particles[i].force[0] = 0;
            particles[i].force[1] = 0;
        }
        forceContactECM(particles);
        forceSelf(particles);
        forceRandom(particles);
        move(particles);
        std::cout << "Step " << step + 1 << " out of " << STEP << std::endl;
    }

    std::ofstream fout1("position10000.txt");
    for (int i = 0; i < NUMBER; i++)
    {
        fout1 << particles[i].position[0] << " " << particles[i].position[1] << " " << particles[i].force[0] << " " << particles[i].force[1] << std::endl;
    }
    fout1.close();
}