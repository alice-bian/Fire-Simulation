////# define OMParallelize

////# ifdef OMParallelize
////# define TOTALThreads 8
////# endif

#include "mac_grid.h"
#include "open_gl_headers.h" 
#include "camera.h"
#include "custom_output.h" 
#include "constants.h" 
#include <math.h>
#include <map>
#include <stdio.h>
#include <functional>
#undef max 
#undef min 
#include <fstream> 

#define  enableSphere false
#define  propogateCentr 24
#define  particleDensity 50


// Globals
MACGrid target;


// NOTE: x -> cols, z -> rows, y -> stacks
MACGrid::RenderMode MACGrid::theRenderMode = SHEETS;
bool MACGrid::theDisplayVel = false;//true

#define FOR_EACH_CELL \
   for(int k = 0; k < theDim[MACGrid::Z]; k++)  \
      for(int j = 0; j < theDim[MACGrid::Y]; j++) \
         for(int i = 0; i < theDim[MACGrid::X]; i++) 

#define FOR_EACH_CELL_REVERSE \
   for(int k = theDim[MACGrid::Z] - 1; k >= 0; k--)  \
      for(int j = theDim[MACGrid::Y] - 1; j >= 0; j--) \
         for(int i = theDim[MACGrid::X] - 1; i >= 0; i--) 

#define FOR_EACH_FACE \
   for(int k = 0; k < theDim[MACGrid::Z]+1; k++) \
      for(int j = 0; j < theDim[MACGrid::Y]+1; j++) \
         for(int i = 0; i < theDim[MACGrid::X]+1; i++) 

#define FOR_EACH_FACE_X \
   for(int k = 0; k < theDim[MACGrid::Z]; k++) \
      for(int j = 0; j < theDim[MACGrid::Y]; j++) \
         for(int i = 0; i < theDim[MACGrid::X]+1; i++)



MACGrid::MACGrid()
{
	initialize();
}

MACGrid::MACGrid(const MACGrid& orig)
{
	mU = orig.mU;
	mV = orig.mV;
	mW = orig.mW;
	mP = orig.mP;
	mD = orig.mD;
	mT = orig.mT;
}

MACGrid& MACGrid::operator=(const MACGrid& orig)
{
	if (&orig == this)
	{
		return *this;
	}
	mU = orig.mU;
	mV = orig.mV;
	mW = orig.mW;
	mP = orig.mP;
	mD = orig.mD;
	mT = orig.mT;

	return *this;
}

MACGrid::~MACGrid()
{
}

void MACGrid::reset()
{
	mU.initialize();
	mV.initialize();
	mW.initialize();
	mP.initialize();
	mD.initialize();
	mT.initialize(0.0);

	calculateAMatrix();
	calculatePreconditioner(AMatrix);
}

void MACGrid::initialize()
{
	reset();
}

void MACGrid::updateSources()
{
	// TODO: Set initial values for density, temperature, velocity
	// STARTED.


	// Simple:
	mV(0, 1, 0) = 2.0;
	mD(0, 0, 0) = 1.0;
	mT(0, 0, 0) = 1.0;
	mV(0, 2, 0) = 2.0;
	mD(0, 1, 0) = 1.0;
	mT(0, 1, 0) = 1.0;



	/*
	double x = theDim[0] - 5;
	double centerY = theDim[1] / 2.0 - 28;
	double centerZ = theDim[2] / 2.0;
	double radius = 3.7;
	for (int y = 0; y < theDim[1]; y++) {
		for (int z = 0; z < theDim[2]; z++) {
			double thisY = y + 0.5;
			double thisZ = z + 0.5;

			double distanceY = thisY - centerY;
			double distanceZ = thisZ - centerZ;
			double distance = sqrt(distanceY * distanceY + distanceZ * distanceZ);

			if (distance < radius) {
				//
				mU(x,y,z) = -10.1;
				mD(x,y,z) = 1.0;
				mT(x,y,z) = 1.0;
				//
				mU(x-1,y,z) = -10.1;
				mD(x-1,y,z) = 1.0;
				mT(x-1,y,z) = 1.0;
				//
				mU(x-2,y,z) = -10.1;
				mD(x-2,y,z) = 1.0;
				mT(x-2,y,z) = 1.0;
				//
				mU(x-3,y,z) = -10.1;
				mD(x-3,y,z) = 1.0;
				mT(x-3,y,z) = 1.0;
			}

		}
	}
	*/


	/*
	double centerX = theDim[0] / 2.0;
	double centerZ = theDim[2] / 2.0;
	double radius = 2.0;
	for (int x = 0; x < theDim[0]; x++) {
		for (int z = 0; z < theDim[2]; z++) {
			double thisX = x + 0.5;
			double thisZ = z + 0.5;

			double distanceX = thisX - centerX;
			double distanceZ = thisZ - centerZ;
			double distance = sqrt(distanceX * distanceX + distanceZ * distanceZ);

			if (distance < radius) {
				mV(x,1,z) = 2.0;
				mD(x,0,z) = 1.0;
				mT(x,0,z) = 1.0;
				//
				mV(x,2,z) = 2.0;
				mD(x,1,z) = 1.0;
				mT(x,1,z) = 1.0;
			}

		}
	}
	 */


	 /*
	 // Center of 3D box:
	 mV(7,1,7) = 2.5;
	 mD(7,0,7) = 1.0;
	 mT(7,0,7) = 1.0;
	 mV(7,2,7) = 2.5;
	 mD(7,1,7) = 1.0;
	 mT(7,1,7) = 1.0;
	 */

	 /*
	 // Hot and cold:
	 //
	 mV(0,1,0) = 3.0;
	 mD(0,1,0) = 1.0;
	 mT(0,1,0) = 1.0;
	 //
	 mV(5,5,0) = -3.0;
	 mD(5,5,0) = 1.0;
	 mT(5,5,0) = 0.0;
	 */

	 /*
	 // Sideways:
	 mU(1,2,0) = 2.0;
	 mD(1,2,0) = 1.0; // Subtract 1???
	 mT(1,2,0) = 1.0; // Subtract 1???
	 */

	 /*
	 // Sideways hot and cold:
	 //
	 mU(1,6,0) = 3.0;
	 mD(0,6,0) = 1.0;
	 mT(0,6,0) = 1.0;
	 //
	 mU(11,6,0) = -3.0;
	 mD(11,6,0) = 1.0;
	 mT(11,6,0) = 0.0;
	 */

	int r = 16;
	FOR_EACH_FACE_X{
		if ((pow((j - propogateCentr + 15), 2) + pow((k - propogateCentr), 2) <= r) && i == 49) {
			vec3 cell_center(theCellSize * (i + 0.5), theCellSize * (j + 0.5), theCellSize * (k + 0.5));
			for (int p = 0; p < particleDensity; p++) {
				double a = ((float)rand() / RAND_MAX - 0.5) * theCellSize;
				double b = ((float)rand() / RAND_MAX - 0.5) * theCellSize;
				double c = ((float)rand() / RAND_MAX - 0.5) * theCellSize;
				vec3 shift(a, b, c);
				vec3 xp = cell_center + shift;
				rendering_particles.push_back(xp);
			}
		}
	}

}

void MACGrid::advectVelocity(double dt)
{
	// TODO: Calculate new velocities and store in target
	// STARTED.
	//target.mU = mU;
	//target.mV = mV;
	//target.mW = mW;



	FOR_EACH_FACE{
		// TODO: Use a loop!
		if (isValidFace(MACGrid::X, i, j, k)) {
			vec3 currentPosition = getFacePosition(MACGrid::X, i, j, k);
			vec3 rewoundPosition = getRewoundPosition(currentPosition, dt);
			vec3 newVelocity = getVelocity(rewoundPosition);
			target.mU(i,j,k) = newVelocity[0];
		}
		if (isValidFace(MACGrid::Y, i, j, k)) {
			vec3 currentPosition = getFacePosition(MACGrid::Y, i, j, k);
			vec3 rewoundPosition = getRewoundPosition(currentPosition, dt);
			vec3 newVelocity = getVelocity(rewoundPosition);
			target.mV(i,j,k) = newVelocity[1];
		}
		if (isValidFace(MACGrid::Z, i, j, k)) {
			vec3 currentPosition = getFacePosition(MACGrid::Z, i, j, k);
			vec3 rewoundPosition = getRewoundPosition(currentPosition, dt);
			vec3 newVelocity = getVelocity(rewoundPosition);
			target.mW(i,j,k) = newVelocity[2];
		}

	}




		// Then save the result to our object
	mU = target.mU;
	mV = target.mV;
	mW = target.mW;

}

void MACGrid::advectTemperature(double dt)
{
	// TODO: Calculate new temp and store in target
	//target.mT = mT;



	FOR_EACH_CELL{
		vec3 currentPosition = getCenter(i,j,k);
		vec3 rewoundPosition = getRewoundPosition(currentPosition, dt);
		double newTemperature = getTemperature(rewoundPosition);
		target.mT(i,j,k) = newTemperature;
	}



		// Then save the result to our object
	mT = target.mT;
}

void MACGrid::advectRenderingParticles(double dt) {

	rendering_particles_vel.resize(rendering_particles.size());
	for (size_t p = 0; p < rendering_particles.size(); p++) {
		vec3 currentPosition = rendering_particles[p];
		vec3 currentVelocity = getVelocity(currentPosition);
		vec3 nextPosition = currentPosition + currentVelocity * dt;
		vec3 clippedNextPosition = clipToGrid(nextPosition, currentPosition);
		// Keep going...
		vec3 nexv = getVelocity(nextPosition);
		vec3 nextVelocity = getVelocity(clippedNextPosition);
		vec3 averageVelocity = (currentVelocity + nextVelocity) / 2.0;
		vec3 betterNextPosition = currentPosition + averageVelocity * dt;
		vec3 clippedBetterNextPosition = clipToGrid(betterNextPosition, currentPosition);
		//====================
		if (enableSphere)
		{
			vec3 vel;
			vec3 centr(sphereC[0] + 1.0, sphereC[1], sphereC[2] + 1.0);
			double r = 3 * theCellSize;
			centr *= theCellSize;
			double radius = Distance(centr, clippedBetterNextPosition);
			if (radius < r) {

				vec3 pos2c = clippedBetterNextPosition - centr;
				pos2c = r * pos2c / radius;
				clippedBetterNextPosition = pos2c + centr;
			}
		}
		//=====================
		rendering_particles[p] = clippedBetterNextPosition;
		rendering_particles_vel[p] = averageVelocity;
	}
}

void MACGrid::advectDensity(double dt)
{
	// TODO: Calculate new densitities and store in target
	// STARTED.
	//target.mD = mD;






	FOR_EACH_CELL{
		vec3 currentPosition = getCenter(i,j,k);
		vec3 rewoundPosition = getRewoundPosition(currentPosition, dt);
		double newDensity = getDensity(rewoundPosition);
		target.mD(i,j,k) = newDensity;
	}









		// Then save the result to our object
	mD = target.mD;
}

void MACGrid::computeBouyancy(double dt)
{
	// TODO: Calculate bouyancy and store in target
	// STARTED.
	//target.mV = mV;




	// Non-negative constants used in buoyancy calculation:
	double alpha = theBuoyancyAlpha; // Gravity's effect on the smoke particles.
	double beta = theBuoyancyBeta; // Buoyancy's effect due to temperature difference.
	// Ambient temperature:
	double ambientTemperature = 0.0;

	FOR_EACH_FACE{

		if (isValidFace(MACGrid::Y, i, j, k)) {
			vec3 position = getFacePosition(MACGrid::Y, i, j, k);
			double temperature = getTemperature(position); // T
			double density = getDensity(position); // s
			double buoyancyForce = -alpha * density + beta * (temperature - ambientTemperature);
			target.mV(i,j,k) = mV(i,j,k) + buoyancyForce;
		}

	}





		// and then save the result to our object
	mV = target.mV;
}

void MACGrid::computeVorticityConfinement(double dt)
{
	// TODO: Calculate vorticity confinement forces
	// Apply the forces to the current velocity and store the result in target
	 // STARTED.

	 // Important:
	target.mU = mU;
	target.mV = mV;
	target.mW = mW;

	GridData tempX; tempX.initialize();
	GridData tempY; tempY.initialize();
	GridData tempZ; tempZ.initialize();
	GridData tempM; tempM.initialize();
	//target.mU.initialize(0.0); //GridData tempFX; tempFX.initialize();
	//target.mV.initialize(0.0); //GridData tempFY; tempFY.initialize();
	//target.mW.initialize(0.0); //GridData tempFZ; tempFZ.initialize();

	double twoDeltaX = (2.0 * theCellSize);

	FOR_EACH_CELL{

		double vorticityX = (mW(i,j + 1,k) - mW(i,j - 1,k)) / twoDeltaX - (mV(i,j,k + 1) - mV(i,j,k - 1)) / twoDeltaX;
		double vorticityY = (mU(i,j,k + 1) - mU(i,j,k - 1)) / twoDeltaX - (mW(i + 1,j,k) - mW(i - 1,j,k)) / twoDeltaX;
		double vorticityZ = (mV(i + 1,j,k) - mV(i - 1,j,k)) / twoDeltaX - (mU(i,j + 1,k) - mU(i,j - 1,k)) / twoDeltaX;
		vec3 vorticity(vorticityX, vorticityY, vorticityZ);

		// Temporarily store the vorticity (as separate components):
		tempX(i,j,k) = vorticityX;
		tempY(i,j,k) = vorticityY;
		tempZ(i,j,k) = vorticityZ;
		// Temporarily store the magnitude (?) of the vorticity, too:
		tempM(i,j,k) = vorticity.Length();
	}

		FOR_EACH_CELL{
			double gradientX = (tempM(i + 1,j,k) - tempM(i - 1,j,k)) / twoDeltaX;
			double gradientY = (tempM(i,j + 1,k) - tempM(i,j - 1,k)) / twoDeltaX;
			double gradientZ = (tempM(i,j,k + 1) - tempM(i,j,k - 1)) / twoDeltaX;
			vec3 gradient(gradientX, gradientY, gradientZ);

			// Normalize the gradient of magnitude of vorticiy:
			vec3 N = gradient / (gradient.Length() + 0.0000000000000000001);

			// Get the stored vorticity:
			vec3 vorticity(tempX(i,j,k), tempY(i,j,k), tempZ(i,j,k));

			// Calculate the confinement force:
			vec3 fConf = theVorticityEpsilon * theCellSize * (N.Cross(vorticity));

			// Spread fConf to the surrounding faces:
			if (isValidFace(0, i,j,k)) target.mU(i,j,k) += fConf[0] / 2.0;
			if (isValidFace(0, i + 1,j,k)) target.mU(i + 1,j,k) += fConf[0] / 2.0;
			if (isValidFace(1, i,j,k)) target.mV(i,j,k) += fConf[1] / 2.0;
			if (isValidFace(1, i,j + 1,k)) target.mV(i,j + 1,k) += fConf[1] / 2.0;
			if (isValidFace(1, i,j,k)) target.mW(i,j,k) += fConf[2] / 2.0;
			if (isValidFace(1, i,j,k + 1)) target.mW(i,j,k + 1) += fConf[2] / 2.0;
	}

		// Then save the result to our object

	mU = target.mU;
	mV = target.mV;
	mW = target.mW;
}

void MACGrid::addExternalForces(double dt)
{
	computeBouyancy(dt);
	computeVorticityConfinement(dt);
}

void MACGrid::project(double dt)
{
	// TODO: Solve Ax = b for pressure
	// 1. Contruct b
	// 2. Construct A 
	// 3. Solve for p
	// Subtract pressure from our velocity and save in target
	 // STARTED.
	 //target.mU = mU;
	 //target.mV = mV;
	 //target.mW = mW;



	 // Divide both sides by this constant, or not:
	double constant = dt / (theAirDensity * (theCellSize * theCellSize));
	//PRINT_LINE( "constant: " << constant );

	// Ax = b
	// Ap = d
	//int numCells = getNumberOfCells();
	//GridDataMatrix A; //boost::numeric::ublas::matrix<double> A(numCells, numCells);
	GridData p; p.initialize(); //boost::numeric::ublas::vector<double> p(numCells);
	GridData d; d.initialize(); //boost::numeric::ublas::vector<double> d(numCells);
	FOR_EACH_CELL{

		//int currentCell = getCellIndex(i,j,k);





		// Construct the matrix A:

		/*
		// Fill the row with 0s:
		for (unsigned Aj = 0; Aj < A.size2(); ++Aj) {
			A(currentCell, Aj) = 0;
		}
		*/
		/*
		// Fill A with 0s:
		for (unsigned Ai = 0; Ai < A.size1(); ++Ai) {
			for (unsigned Aj = 0; Aj < A.size2(); ++Aj) {
				A(Ai, Aj) = 0;
			}
		}
		*/

		/*
		int cellLowX = getCellIndex(i-1,j,k);
		int cellHighX = getCellIndex(i+1,j,k);
		int cellLowY = getCellIndex(i,j-1,k);
		int cellHighY = getCellIndex(i,j+1,k);
		int cellLowZ = getCellIndex(i,j,k-1);
		int cellHighZ = getCellIndex(i,j,k+1);
		*/
		/*
		int numFluidNeighbors = 0;
		if (i-1 >= 0) {
			A.plusI(i-1,j,k) = -1;//A.minusI(i,j,k) = -1;//A(currentCell, cellLowX) = -1;
			numFluidNeighbors++;
		}
		if (i+1 < theDim[MACGrid::X]) {
			A.plusI(i,j,k) = -1;//A(currentCell, cellHighX) = -1;
			numFluidNeighbors++;
		}
		if (j-1 >= 0) {
			A.plusJ(i,j-1,k) = -1;//A.minusJ(i,j,k) = -1;//A(currentCell, cellLowY) = -1;
			numFluidNeighbors++;
		}
		if (j+1 < theDim[MACGrid::Y]) {
			A.plusJ(i,j,k) = -1;//A(currentCell, cellHighY) = -1;
			numFluidNeighbors++;
		}
		if (k-1 >= 0) {
			A.plusK(i,j,k-1) = -1;//A.minusK(i,j,k) = -1;//A(currentCell, cellLowZ) = -1;
			numFluidNeighbors++;
		}
		if (k+1 < theDim[MACGrid::Z]) {
			A.plusK(i,j,k) = -1;//A(currentCell, cellHighZ) = -1;
			numFluidNeighbors++;
		}
		// Set the diagonal:
		A.diag(i,j,k) = numFluidNeighbors;//A(currentCell, currentCell) = numFluidNeighbors;//6
		//if (i == 0) A(currentCell, currentCell)--; // TEMP!!!!!!
		//if (j+1 == theDim[MACGrid::Y]) A(currentCell, currentCell)--; // TEMP!!!!!!
		//if (k+1 == theDim[MACGrid::Z]) A(currentCell, currentCell)--; // TEMP!!!!!!
		*/





		// Construct the vector of divergences d:
		double velLowX = mU(i,j,k);
		double velHighX = mU(i + 1,j,k);
		double velLowY = mV(i,j,k);
		double velHighY = mV(i,j + 1,k);
		double velLowZ = mW(i,j,k);
		double velHighZ = mW(i,j,k + 1);
		// Use 0 for solid boundary velocities:
		if (i == 0) velLowX = 0;
		if (i + 1 == theDim[MACGrid::X]) velHighX = 0;
		if (j == 0) velLowY = 0;
		if (j + 1 == theDim[MACGrid::Y]) velHighY = 0;
		if (k == 0) velLowZ = 0;
		if (k + 1 == theDim[MACGrid::Z]) velHighZ = 0;
		double divergence = ((velHighX - velLowX) + (velHighY - velLowY) + (velHighZ - velLowZ)) / theCellSize;
		d(i,j,k) = -divergence;//d(currentCell) = -divergence;
		/*
		PRINT_LINE( "CELL " << currentCell << " (" << i << ", " << j << ", " << k << ")" );
		PRINT_LINE( "Low Y: " << velLowY );
		PRINT_LINE( "High Y: " << velHighY );
		PRINT_LINE( "Divergence: " << divergence );
		PRINT_LINE( endl );
		*/

	}

		/*
		PRINT_LINE( "MATRIX A: " );
		for (unsigned Ai = 0; Ai < A.size1(); ++Ai) {
			for (unsigned Aj = 0; Aj < A.size2(); ++Aj) {
				PRINT( A(Ai,Aj) << "\t" );
			}
			PRINT( endl );
		}
		PRINT( endl );
		*/
		/*
		PRINT_LINE( "VECTOR d: " );
		for (unsigned di = 0; di < d.size(); ++di) {
			PRINT_LINE( d(di) );
		}
		PRINT( endl );
		*/




		// Solve for p:
	bool converged = preconditionedConjugateGradient(AMatrix, p, d, 150, 0.01);//bool converged = cg_solve(A, d, p, 150, 0.01);
	if (converged == false) {
		PRINT_LINE("BAD!");
	}

	// Multiply the computed pressures by the constant, instead of dividing the matrix entries by it.
	FOR_EACH_CELL{//for (unsigned pi = 0; pi < p.size(); ++pi) {
		p(i,j,k) /= constant;//p(pi) /= constant;
	}

		/*
		PRINT_LINE( "PRESSURE p: " );
		for (unsigned pi = 0; pi < p.size(); ++pi) {
			PRINT_LINE( p(pi) );
		}
		PRINT_LINE( endl );
		*/

		// Save the pressure:
		FOR_EACH_CELL{
		//int currentCell = getCellIndex(i,j,k);
		target.mP(i,j,k) = p(i,j,k);//p(currentCell);
	}
		FOR_EACH_FACE{ // ????????????????????????????????????????????????????????
			/*
			// Different than above:
			int cellLowX = getCellIndex(i-1,j,k);
			int cellHighX = getCellIndex(i,j,k);
			int cellLowY = getCellIndex(i,j-1,k);
			int cellHighY = getCellIndex(i,j,k);
			int cellLowZ = getCellIndex(i,j,k-1);
			int cellHighZ = getCellIndex(i,j,k);
			*/

			// Initialize the pressure to 0.
			double pLowX = 0;
			double pHighX = 0;
			double pLowY = 0;
			double pHighY = 0;
			double pLowZ = 0;
			double pHighZ = 0;

			double solidBoundaryConstant = theAirDensity * theCellSize / dt;
			if (isValidFace(MACGrid::X, i, j, k)) {
				if (i - 1 >= 0) {
					pLowX = p(i - 1,j,k);//p(cellLowX);
				}

				if (i < theDim[MACGrid::X]) {
					pHighX = p(i,j,k);//p(cellHighX);
				}
	 else {

  }

  if (i - 1 < 0) {
	  pLowX = pHighX - solidBoundaryConstant * (mU(i,j,k) - 0);
  }

  if (i >= theDim[MACGrid::X]) {
	  pHighX = pLowX + solidBoundaryConstant * (mU(i,j,k) - 0);
  }

}
if (isValidFace(MACGrid::Y, i, j, k)) {
	if (j - 1 >= 0) {
		pLowY = p(i,j - 1,k);//p(cellLowY);
	}

	if (j < theDim[MACGrid::Y]) {
		pHighY = p(i,j,k);//p(cellHighY);
	}

	if (j - 1 < 0) {
		pLowY = pHighY - solidBoundaryConstant * (mV(i,j,k) - 0);
	}

	if (j >= theDim[MACGrid::Y]) {
		pHighY = pLowY + solidBoundaryConstant * (mV(i,j,k) - 0);
	}
}
if (isValidFace(MACGrid::Z, i, j, k)) {
	if (k - 1 >= 0) {
		pLowZ = p(i,j,k - 1);//p(cellLowZ);
	}

	if (k < theDim[MACGrid::Z]) {
		pHighZ = p(i,j,k);//p(cellHighZ);
	}

	if (k - 1 < 0) {
		pLowZ = pHighZ - solidBoundaryConstant * (mW(i,j,k) - 0);
	}

	if (k >= theDim[MACGrid::Z]) {
		pHighZ = pLowZ + solidBoundaryConstant * (mW(i,j,k) - 0);
	}
}
// Update the velocities:
double anotherConstant = dt / theAirDensity; // Bottom of page 27.
if (isValidFace(MACGrid::X, i, j, k)) {
	target.mU(i,j,k) = mU(i,j,k) - anotherConstant * (pHighX - pLowX) / theCellSize;
}
if (isValidFace(MACGrid::Y, i, j, k)) {
	target.mV(i,j,k) = mV(i,j,k) - anotherConstant * (pHighY - pLowY) / theCellSize;
}
if (isValidFace(MACGrid::Z, i, j, k)) {
	target.mW(i,j,k) = mW(i,j,k) - anotherConstant * (pHighZ - pLowZ) / theCellSize;
}
	}


#ifdef _DEBUG
		// Check border velocities:
		FOR_EACH_FACE{
			if (isValidFace(MACGrid::X, i, j, k)) {

				if (i == 0) {
					if (abs(target.mU(i,j,k)) > 0.0000001) {
						PRINT_LINE("LOW X:  " << target.mU(i,j,k));
						//target.mU(i,j,k) = 0;
					}
				}

				if (i == theDim[MACGrid::X]) {
					if (abs(target.mU(i,j,k)) > 0.0000001) {
						PRINT_LINE("HIGH X: " << target.mU(i,j,k));
						//target.mU(i,j,k) = 0;
					}
				}

			}
			if (isValidFace(MACGrid::Y, i, j, k)) {


				if (j == 0) {
					if (abs(target.mV(i,j,k)) > 0.0000001) {
						PRINT_LINE("LOW Y:  " << target.mV(i,j,k));
						//target.mV(i,j,k) = 0;
					}
				}

				if (j == theDim[MACGrid::Y]) {
					if (abs(target.mV(i,j,k)) > 0.0000001) {
						PRINT_LINE("HIGH Y: " << target.mV(i,j,k));
						//target.mV(i,j,k) = 0;
					}
				}

			}
			if (isValidFace(MACGrid::Z, i, j, k)) {

				if (k == 0) {
					if (abs(target.mW(i,j,k)) > 0.0000001) {
						PRINT_LINE("LOW Z:  " << target.mW(i,j,k));
						//target.mW(i,j,k) = 0;
					}
				}

				if (k == theDim[MACGrid::Z]) {
					if (abs(target.mW(i,j,k)) > 0.0000001) {
						PRINT_LINE("HIGH Z: " << target.mW(i,j,k));
						//target.mW(i,j,k) = 0;
					}
				}
			}
	}
#endif


		// Then save the result to our object
	mP = target.mP;
	mU = target.mU;
	mV = target.mV;
	mW = target.mW;




#ifdef _DEBUG
	// IMPLEMENT THIS AS A SANITY CHECK: assert (checkDivergence());
	// TODO: Fix duplicate code:
	FOR_EACH_CELL{
		// Construct the vector of divergences d:
		 double velLowX = mU(i,j,k);
		 double velHighX = mU(i + 1,j,k);
		 double velLowY = mV(i,j,k);
		 double velHighY = mV(i,j + 1,k);
		 double velLowZ = mW(i,j,k);
		 double velHighZ = mW(i,j,k + 1);
		 double divergence = ((velHighX - velLowX) + (velHighY - velLowY) + (velHighZ - velLowZ)) / theCellSize;
		 if (abs(divergence) > 0.02) {
			 PRINT_LINE("WARNING: Divergent! ");
			 PRINT_LINE("Divergence: " << divergence);
			 PRINT_LINE("Cell: " << i << ", " << j << ", " << k);
		 }
	}
#endif


}

vec3 MACGrid::getVelocity(const vec3& pt)
{
	vec3 vel;
	vel[0] = getVelocityX(pt);
	vel[1] = getVelocityY(pt);
	vel[2] = getVelocityZ(pt);
	return vel;
}

double MACGrid::getVelocityX(const vec3& pt)
{
	return mU.interpolate(pt);
}

double MACGrid::getVelocityY(const vec3& pt)
{
	return mV.interpolate(pt);
}

double MACGrid::getVelocityZ(const vec3& pt)
{
	return mW.interpolate(pt);
}

double MACGrid::getTemperature(const vec3& pt)
{
	return mT.interpolate(pt);
}

double MACGrid::getDensity(const vec3& pt)
{
	return mD.interpolate(pt);
}

vec3 MACGrid::getCenter(int i, int j, int k)
{
	double xstart = theCellSize / 2.0;
	double ystart = theCellSize / 2.0;
	double zstart = theCellSize / 2.0;

	double x = xstart + i * theCellSize;
	double y = ystart + j * theCellSize;
	double z = zstart + k * theCellSize;
	return vec3(x, y, z);
}


vec3 MACGrid::getRewoundPosition(const vec3 & currentPosition, const double dt) {

	/*
	// EULER (RK1):
	vec3 currentVelocity = getVelocity(currentPosition);
	vec3 rewoundPosition = currentPosition - currentVelocity * dt;
	vec3 clippedRewoundPosition = clipToGrid(rewoundPosition, currentPosition);
	return clippedRewoundPosition;
	*/

	// HEUN / MODIFIED EULER (RK2):
	vec3 currentVelocity = getVelocity(currentPosition);
	vec3 rewoundPosition = currentPosition - currentVelocity * dt;
	vec3 clippedRewoundPosition = clipToGrid(rewoundPosition, currentPosition);
	// Keep going...
	vec3 rewoundVelocity = getVelocity(clippedRewoundPosition);
	vec3 averageVelocity = (currentVelocity + rewoundVelocity) / 2.0;
	vec3 betterRewoundPosition = currentPosition - averageVelocity * dt;
	vec3 clippedBetterRewoundPosition = clipToGrid(betterRewoundPosition, currentPosition);
	return clippedBetterRewoundPosition;

}


vec3 MACGrid::clipToGrid(const vec3& outsidePoint, const vec3& insidePoint) {
	/*
	// OLD:
	vec3 rewindPosition = outsidePoint;
	if (rewindPosition[0] < 0) rewindPosition[0] = 0; // TEMP!
	if (rewindPosition[1] < 0) rewindPosition[1] = 0; // TEMP!
	if (rewindPosition[2] < 0) rewindPosition[2] = 0; // TEMP!
	if (rewindPosition[0] > theDim[MACGrid::X]) rewindPosition[0] = theDim[MACGrid::X]; // TEMP!
	if (rewindPosition[1] > theDim[MACGrid::Y]) rewindPosition[1] = theDim[MACGrid::Y]; // TEMP!
	if (rewindPosition[2] > theDim[MACGrid::Z]) rewindPosition[2] = theDim[MACGrid::Z]; // TEMP!
	return rewindPosition;
	*/

	vec3 clippedPoint = outsidePoint;

	for (int i = 0; i < 3; i++) {
		if (clippedPoint[i] < 0) {
			vec3 distance = clippedPoint - insidePoint;
			double newDistanceI = 0 - insidePoint[i];
			double ratio = newDistanceI / distance[i];
			clippedPoint = insidePoint + distance * ratio;
		}
		if (clippedPoint[i] > getSize(i)) {
			vec3 distance = clippedPoint - insidePoint;
			double newDistanceI = getSize(i) - insidePoint[i];
			double ratio = newDistanceI / distance[i];
			clippedPoint = insidePoint + distance * ratio;
		}
	}

#ifdef _DEBUG
	// Make sure the point is now in the grid:
	if (clippedPoint[0] < 0 || clippedPoint[1] < 0 || clippedPoint[2] < 0 || clippedPoint[0] > getSize(0) || clippedPoint[1] > getSize(1) || clippedPoint[2] > getSize(2)) {
		//PRINT_LINE("WARNING: Clipped point is outside grid!");
	}
#endif

	return clippedPoint;

}


double MACGrid::getSize(int dimension) {
	return theDim[dimension] * theCellSize;
}


int MACGrid::getCellIndex(int i, int j, int k)
{
	return i + j * theDim[MACGrid::X] + k * theDim[MACGrid::Y] * theDim[MACGrid::X];
}


int MACGrid::getNumberOfCells()
{
	return theDim[MACGrid::X] * theDim[MACGrid::Y] * theDim[MACGrid::Z];
}


bool MACGrid::isValidCell(int i, int j, int k)
{
	if (i >= theDim[MACGrid::X] || j >= theDim[MACGrid::Y] || k >= theDim[MACGrid::Z]) {
		return false;
	}

	if (i < 0 || j < 0 || k < 0) {
		return false;
	}

	return true;
}


bool MACGrid::isValidFace(int dimension, int i, int j, int k)
{
	if (dimension == 0) {
		if (i > theDim[MACGrid::X] || j >= theDim[MACGrid::Y] || k >= theDim[MACGrid::Z]) {
			return false;
		}
	}
	else if (dimension == 1) {
		if (i >= theDim[MACGrid::X] || j > theDim[MACGrid::Y] || k >= theDim[MACGrid::Z]) {
			return false;
		}
	}
	else if (dimension == 2) {
		if (i >= theDim[MACGrid::X] || j >= theDim[MACGrid::Y] || k > theDim[MACGrid::Z]) {
			return false;
		}
	}

	if (i < 0 || j < 0 || k < 0) {
		return false;
	}

	return true;
}


vec3 MACGrid::getFacePosition(int dimension, int i, int j, int k)
{
	if (dimension == 0) {
		return vec3(i * theCellSize, (j + 0.5) * theCellSize, (k + 0.5) * theCellSize);
	}
	else if (dimension == 1) {
		return vec3((i + 0.5) * theCellSize, j * theCellSize, (k + 0.5) * theCellSize);
	}
	else if (dimension == 2) {
		return vec3((i + 0.5) * theCellSize, (j + 0.5) * theCellSize, k * theCellSize);
	}

	return vec3(0, 0, 0); //???

}

void MACGrid::calculateAMatrix() {

	FOR_EACH_CELL{

		int numFluidNeighbors = 0;
		if (i - 1 >= 0) {
			AMatrix.plusI(i - 1,j,k) = -1;
			numFluidNeighbors++;
		}
		if (i + 1 < theDim[MACGrid::X]) {
			AMatrix.plusI(i,j,k) = -1;
			numFluidNeighbors++;
		}
		if (j - 1 >= 0) {
			AMatrix.plusJ(i,j - 1,k) = -1;
			numFluidNeighbors++;
		}
		if (j + 1 < theDim[MACGrid::Y]) {
			AMatrix.plusJ(i,j,k) = -1;
			numFluidNeighbors++;
		}
		if (k - 1 >= 0) {
			AMatrix.plusK(i,j,k - 1) = -1;
			numFluidNeighbors++;
		}
		if (k + 1 < theDim[MACGrid::Z]) {
			AMatrix.plusK(i,j,k) = -1;
			numFluidNeighbors++;
		}
		// Set the diagonal:
		AMatrix.diag(i,j,k) = numFluidNeighbors;
	}
}


bool MACGrid::preconditionedConjugateGradient(const GridDataMatrix & A, GridData & p, const GridData & d, int maxIterations, double tolerance) {
	// Solves Ap = d for p.

	FOR_EACH_CELL{
		p(i,j,k) = 0.0; // Initial guess p = 0.	
	}

	GridData r = d; // Residual vector.

	/*
	PRINT_LINE("r: ");
	FOR_EACH_CELL {
		PRINT_LINE(r(i,j,k));
	}
	*/
	GridData z; z.initialize();
	applyPreconditioner(r, A, z); // Auxillary vector.
	/*
	PRINT_LINE("z: ");
	FOR_EACH_CELL {
		PRINT_LINE(z(i,j,k));
	}
	*/

	GridData s = z; // Search vector;

	double sigma = dotProduct(z, r);

	for (int iteration = 0; iteration < maxIterations; iteration++) {

		double rho = sigma; // According to TA. Here???

		apply(A, s, z); // z = applyA(s);

		double alpha = rho / dotProduct(z, s);

		GridData alphaTimesS; alphaTimesS.initialize();
		multiply(alpha, s, alphaTimesS);
		add(p, alphaTimesS, p);
		//p += alpha * s;

		GridData alphaTimesZ; alphaTimesZ.initialize();
		multiply(alpha, z, alphaTimesZ);
		subtract(r, alphaTimesZ, r);
		//r -= alpha * z;

		if (maxMagnitude(r) <= tolerance) {
			//PRINT_LINE("PCG converged in " << (iteration + 1) << " iterations.");
			return true; //return p;
		}

		applyPreconditioner(r, A, z); // z = applyPreconditioner(r);

		double sigmaNew = dotProduct(z, r);

		double beta = sigmaNew / rho;

		GridData betaTimesS; betaTimesS.initialize();
		multiply(beta, s, betaTimesS);
		add(z, betaTimesS, s);
		//s = z + beta * s;

		sigma = sigmaNew;
	}

	PRINT_LINE("PCG didn't converge!");
	return false;

}


void MACGrid::calculatePreconditioner(const GridDataMatrix & A) {

	precon.initialize();

	// All assuming GridData () operator returns 0 with invalid inputs.

	// CALCULATE THE PRECONDITIONER:

	double tau = 0.97; // Tuning constant.
	FOR_EACH_CELL{
		//if (A.diag(i,j,k) != 0.0) { // If cell is a fluid...
			double e = A.diag(i,j,k) - pow(A.plusI(i - 1,j,k) * precon(i - 1,j,k), 2)
										- pow(A.plusJ(i,j - 1,k) * precon(i,j - 1,k), 2)
										- pow(A.plusK(i,j,k - 1) * precon(i,j,k - 1), 2)
						- tau * (A.plusI(i - 1,j,k) * (A.plusJ(i - 1,j,k) + A.plusK(i - 1,j,k)) * pow(precon(i - 1,j,k), 2)
										+ A.plusJ(i,j - 1,k) * (A.plusI(i,j - 1,k) + A.plusK(i,j - 1,k)) * pow(precon(i,j - 1,k), 2)
										+ A.plusK(i,j,k - 1) * (A.plusI(i,j,k - 1) + A.plusJ(i,j,k - 1)) * pow(precon(i,j,k - 1), 2)
										);
			precon(i,j,k) = 1.0 / sqrt(e + 0.00000000000000000000000000001);
			//}
	}

		/*
		FOR_EACH_CELL {
			PRINT_LINE(AMatrix.diag(i,j,k));
			PRINT_LINE(precon(i,j,k));
		}
		*/

}


void MACGrid::applyPreconditioner(const GridData & r, const GridDataMatrix & A, GridData & z) {

	/*
	// TEST: Bypass preconditioner:
	z = r;
	return;
	*/


	// APPLY THE PRECONDITIONER:

	// Solve Lq = r for q:

	GridData q;
	q.initialize();

	FOR_EACH_CELL{
		//if (A.diag(i,j,k) != 0.0) { // If cell is a fluid.
			double t = r(i,j,k) - A.plusI(i - 1,j,k) * precon(i - 1,j,k) * q(i - 1,j,k)
									- A.plusJ(i,j - 1,k) * precon(i,j - 1,k) * q(i,j - 1,k)
									- A.plusK(i,j,k - 1) * precon(i,j,k - 1) * q(i,j,k - 1);
			q(i,j,k) = t * precon(i,j,k);
			//}
	}

		// Solve L^Tz = q for z:

		FOR_EACH_CELL_REVERSE{
		//if (A.diag(i,j,k) != 0.0) { // If cell is a fluid.
			double t = q(i,j,k) - A.plusI(i,j,k) * precon(i,j,k) * z(i + 1,j,k)
									- A.plusJ(i,j,k) * precon(i,j,k) * z(i,j + 1,k)
									- A.plusK(i,j,k) * precon(i,j,k) * z(i,j,k + 1);
			z(i,j,k) = t * precon(i,j,k);
			//}
	}

}



double MACGrid::dotProduct(const GridData & vector1, const GridData & vector2) {

	double result = 0.0;

	FOR_EACH_CELL{
		result += vector1(i,j,k) * vector2(i,j,k);
	}

	return result;
}


void MACGrid::add(const GridData & vector1, const GridData & vector2, GridData & result) {

	FOR_EACH_CELL{
		result(i,j,k) = vector1(i,j,k) + vector2(i,j,k);
	}

}


void MACGrid::subtract(const GridData & vector1, const GridData & vector2, GridData & result) {

	FOR_EACH_CELL{
		result(i,j,k) = vector1(i,j,k) - vector2(i,j,k);
	}

}


void MACGrid::multiply(const double scalar, const GridData & vector, GridData & result) {

	FOR_EACH_CELL{
		result(i,j,k) = scalar * vector(i,j,k);
	}

}


double MACGrid::maxMagnitude(const GridData & vector) {

	double result = 0.0;

	FOR_EACH_CELL{
		if (abs(vector(i,j,k)) > result) result = abs(vector(i,j,k));
	}

	return result;
}


void MACGrid::apply(const GridDataMatrix & matrix, const GridData & vector, GridData & result) {

	FOR_EACH_CELL{ // For each row of the matrix.

		double diag = 0;
		double plusI = 0;
		double plusJ = 0;
		double plusK = 0;
		double minusI = 0;
		double minusJ = 0;
		double minusK = 0;

		diag = matrix.diag(i,j,k) * vector(i,j,k);
		if (isValidCell(i + 1,j,k)) plusI = matrix.plusI(i,j,k) * vector(i + 1,j,k);
		if (isValidCell(i,j + 1,k)) plusJ = matrix.plusJ(i,j,k) * vector(i,j + 1,k);
		if (isValidCell(i,j,k + 1)) plusK = matrix.plusK(i,j,k) * vector(i,j,k + 1);
		if (isValidCell(i - 1,j,k)) minusI = matrix.plusI(i - 1,j,k) * vector(i - 1,j,k);
		if (isValidCell(i,j - 1,k)) minusJ = matrix.plusJ(i,j - 1,k) * vector(i,j - 1,k);
		if (isValidCell(i,j,k - 1)) minusK = matrix.plusK(i,j,k - 1) * vector(i,j,k - 1);

		result(i,j,k) = diag + plusI + plusJ + plusK + minusI + minusJ + minusK;
	}

}

void MACGrid::saveParticle(std::string filename) {
	// Generate quad mesh (without faces) for rendering.
	std::ofstream smoke_quad_mesh;
	smoke_quad_mesh.open(filename);
	// Add vertices
	for (int i = 0; i < rendering_particles.size(); i++) {
		smoke_quad_mesh << "v";
		for (int j = 0; j < 3; j++) {
			smoke_quad_mesh << " " << rendering_particles[i][j];
		}
		smoke_quad_mesh << "\n";
	}
	smoke_quad_mesh.close();
}

//save colors to obj files 
void MACGrid::saveColors(std::string filename) {

	ofstream fout(filename); // open the file 
	for (unsigned int i = 0; i < rendering_colors.size(); i++)
	{
		vec3 p;
		//std::vector<float> p;
		for (int k = 0; k < 3; k++)
		{
			//p.push_back(rendering_particles[i][k]);
			p[k] = rendering_colors[i][k];
		}
		//add line to file 
		fout << "v " << p[0] << " " << p[1] << " " << p[2] << endl;
	}
	fout.close(); //close the file 

}

void MACGrid::saveSmoke(const char* fileName) {
	std::ofstream fileOut(fileName);
	if (fileOut.is_open()) {
		FOR_EACH_CELL{
			fileOut << mD(i,j,k) << std::endl;
		}
		fileOut.close();
	}
}

void MACGrid::draw(const Camera& c)
{
	drawWireGrid();
	if (theDisplayVel) drawVelocities();
	drawParticles();
	//drawSmokeCubes(c); 
	//if (theRenderMode == CUBES) drawSmokeCubes(c);
	//else drawSmoke(c);
}

void MACGrid::drawParticles() {
	rendering_particles = {}; //empty this for every frame 
	rendering_colors = {}; 
	FOR_EACH_CELL
	{

	vec3 cellCenter = getCenter(i, j, k);
	vec4 color = getRenderColor(i, j, k);
	if (!(color[3] > 0.01)) {
		continue;
	}

	//randomly place
	int particle_density = 30;
	for (int p = 0; p < particle_density; p++) {
		double a = ((float)rand() / RAND_MAX - 0.5) * theCellSize;
		double b = ((float)rand() / RAND_MAX - 0.5) * theCellSize;
		double c = ((float)rand() / RAND_MAX - 0.5) * theCellSize;
		vec3 shift(a, b, c);
		vec3 xp = cellCenter + shift;
		double density = getDensity(xp);
		//vec4 col = getRenderColor(xp);
		//TODO LATER: take temperature and density and LERPing into account (hot, cold colors) when setting the colors 
		//create a vec3 color based on the density value (higher density = darker color)
		vec3 baseColor = vec3(237.0, 63.0, 14.0) / 255.0;
		//multiply it by the density
		baseColor *= density;
		rendering_colors.push_back(baseColor);

		rendering_particles.push_back(xp);
	}

	}
}

void MACGrid::drawVelocities()
{
	// draw line at each center
	//glColor4f(0.0, 1.0, 0.0, 1.0);
	glBegin(GL_LINES);
	FOR_EACH_CELL
	{
	   vec3 pos = getCenter(i,j,k);
	   vec3 vel = getVelocity(pos);
	   if (vel.Length() > 0.0001)
	   {
		   //vel.Normalize(); 
		   vel *= theCellSize / 2.0;
		   vel += pos;
		   glColor4f(1.0, 1.0, 0.0, 1.0);
		   glVertex3dv(pos.n);
		   glColor4f(0.0, 1.0, 0.0, 1.0);
		   glVertex3dv(vel.n);
		 }
	}
	glEnd();
}

vec4 MACGrid::getRenderColor(int i, int j, int k)
{

	double value = mD(i, j, k);
	vec4 coldColor(0.5, 0.5, 1.0, value);
	vec4 hotColor(1.0, 0.5, 0.5, value);
	return LERP(coldColor, hotColor, mT(i, j, k));


	/*
	// OLD:
	double value = mD(i, j, k);
	return vec4(1.0, 0.9, 1.0, value);
	*/
}

vec4 MACGrid::getRenderColor(const vec3& pt)
{
	double value = getDensity(pt);
	vec4 coldColor(0.5, 0.5, 1.0, value);
	vec4 hotColor(1.0, 0.5, 0.5, value);
	return LERP(coldColor, hotColor, getTemperature(pt));

	/*
	// OLD:
	double value = getDensity(pt);
	return vec4(1.0, 1.0, 1.0, value);
	*/
}

void MACGrid::drawZSheets(bool backToFront)
{
	// Draw K Sheets from back to front
	double back = (theDim[2])*theCellSize;
	double top = (theDim[1])*theCellSize;
	double right = (theDim[0])*theCellSize;

	double stepsize = theCellSize * 0.25;

	double startk = back - stepsize;
	double endk = 0;
	double stepk = -theCellSize;

	if (!backToFront)
	{
		startk = 0;
		endk = back;
		stepk = theCellSize;
	}

	for (double k = startk; backToFront ? k > endk : k < endk; k += stepk)
	{
		for (double j = 0.0; j < top; )
		{
			glBegin(GL_QUAD_STRIP);
			for (double i = 0.0; i <= right; i += stepsize)
			{
				vec3 pos1 = vec3(i, j, k);
				vec3 pos2 = vec3(i, j + stepsize, k);

				vec4 color1 = getRenderColor(pos1);
				vec4 color2 = getRenderColor(pos2);

				glColor4dv(color1.n);
				glVertex3dv(pos1.n);

				glColor4dv(color2.n);
				glVertex3dv(pos2.n);
			}
			glEnd();
			j += stepsize;

			glBegin(GL_QUAD_STRIP);
			for (double i = right; i >= 0.0; i -= stepsize)
			{
				vec3 pos1 = vec3(i, j, k);
				vec3 pos2 = vec3(i, j + stepsize, k);

				vec4 color1 = getRenderColor(pos1);
				vec4 color2 = getRenderColor(pos2);

				glColor4dv(color1.n);
				glVertex3dv(pos1.n);

				glColor4dv(color2.n);
				glVertex3dv(pos2.n);
			}
			glEnd();
			j += stepsize;
		}
	}
}

void MACGrid::drawXSheets(bool backToFront)
{
	// Draw K Sheets from back to front
	double back = (theDim[2])*theCellSize;
	double top = (theDim[1])*theCellSize;
	double right = (theDim[0])*theCellSize;

	double stepsize = theCellSize * 0.25;

	double starti = right - stepsize;
	double endi = 0;
	double stepi = -theCellSize;

	if (!backToFront)
	{
		starti = 0;
		endi = right;
		stepi = theCellSize;
	}

	for (double i = starti; backToFront ? i > endi : i < endi; i += stepi)
	{
		for (double j = 0.0; j < top; )
		{
			glBegin(GL_QUAD_STRIP);
			for (double k = 0.0; k <= back; k += stepsize)
			{
				vec3 pos1 = vec3(i, j, k);
				vec3 pos2 = vec3(i, j + stepsize, k);

				vec4 color1 = getRenderColor(pos1);
				vec4 color2 = getRenderColor(pos2);

				glColor4dv(color1.n);
				glVertex3dv(pos1.n);

				glColor4dv(color2.n);
				glVertex3dv(pos2.n);
			}
			glEnd();
			j += stepsize;

			glBegin(GL_QUAD_STRIP);
			for (double k = back; k >= 0.0; k -= stepsize)
			{
				vec3 pos1 = vec3(i, j, k);
				vec3 pos2 = vec3(i, j + stepsize, k);

				vec4 color1 = getRenderColor(pos1);
				vec4 color2 = getRenderColor(pos2);

				glColor4dv(color1.n);
				glVertex3dv(pos1.n);

				glColor4dv(color2.n);
				glVertex3dv(pos2.n);
			}
			glEnd();
			j += stepsize;
		}
	}
}


void MACGrid::drawSmoke(const Camera& c)
{
	vec3 eyeDir = c.getBackward();
	double zresult = fabs(Dot(eyeDir, vec3(1, 0, 0)));
	double xresult = fabs(Dot(eyeDir, vec3(0, 0, 1)));
	//double yresult = fabs(Dot(eyeDir, vec3(0,1,0)));

	if (zresult < xresult)
	{
		drawZSheets(c.getPosition()[2] < 0);
	}
	else
	{
		drawXSheets(c.getPosition()[0] < 0);
	}
}

void MACGrid::drawSmokeCubes(const Camera& c)
{
	std::multimap<double, MACGrid::Cube, std::greater<double> > cubes;
	FOR_EACH_CELL
	{
	   MACGrid::Cube cube;
	   cube.color = getRenderColor(i,j,k);
	   cube.pos = getCenter(i,j,k);
	   cube.dist = DistanceSqr(cube.pos, c.getPosition());
	   cubes.insert(make_pair(cube.dist, cube));
	}

		// Draw cubes from back to front
	std::multimap<double, MACGrid::Cube, std::greater<double> >::const_iterator it;
	for (it = cubes.begin(); it != cubes.end(); ++it)
	{
		drawCube(it->second);
	}
}

void MACGrid::drawWireGrid()
{
	// Display grid in light grey, draw top & bottom

	double xstart = 0.0;
	double ystart = 0.0;
	double zstart = 0.0;
	double xend = theDim[0] * theCellSize;
	double yend = theDim[1] * theCellSize;
	double zend = theDim[2] * theCellSize;

	glPushAttrib(GL_LIGHTING_BIT | GL_LINE_BIT);
	glDisable(GL_LIGHTING);
	glColor3f(0.25, 0.25, 0.25);

	glBegin(GL_LINES);
	for (int i = 0; i <= theDim[0]; i++)
	{
		double x = xstart + i * theCellSize;
		glVertex3d(x, ystart, zstart);
		glVertex3d(x, ystart, zend);

		glVertex3d(x, yend, zstart);
		glVertex3d(x, yend, zend);
	}

	for (int i = 0; i <= theDim[2]; i++)
	{
		double z = zstart + i * theCellSize;
		glVertex3d(xstart, ystart, z);
		glVertex3d(xend, ystart, z);

		glVertex3d(xstart, yend, z);
		glVertex3d(xend, yend, z);
	}

	glVertex3d(xstart, ystart, zstart);
	glVertex3d(xstart, yend, zstart);

	glVertex3d(xend, ystart, zstart);
	glVertex3d(xend, yend, zstart);

	glVertex3d(xstart, ystart, zend);
	glVertex3d(xstart, yend, zend);

	glVertex3d(xend, ystart, zend);
	glVertex3d(xend, yend, zend);
	glEnd();
	glPopAttrib();

	glEnd();
}

#define LEN 0.5
void MACGrid::drawFace(const MACGrid::Cube& cube)
{
	glColor4dv(cube.color.n);
	glPushMatrix();
	glTranslated(cube.pos[0], cube.pos[1], cube.pos[2]);
	glScaled(theCellSize, theCellSize, theCellSize);
	glBegin(GL_QUADS);
	glNormal3d(0.0, 0.0, 1.0);
	glVertex3d(-LEN, -LEN, LEN);
	glVertex3d(-LEN, LEN, LEN);
	glVertex3d(LEN, LEN, LEN);
	glVertex3d(LEN, -LEN, LEN);
	glEnd();
	glPopMatrix();
}

void MACGrid::drawCube(const MACGrid::Cube& cube)
{
	glColor4dv(cube.color.n);
	glPushMatrix();
	glTranslated(cube.pos[0], cube.pos[1], cube.pos[2]);
	glScaled(theCellSize, theCellSize, theCellSize);
	glBegin(GL_QUADS);
	glNormal3d(0.0, -1.0, 0.0);
	glVertex3d(-LEN, -LEN, -LEN);
	glVertex3d(-LEN, -LEN, LEN);
	glVertex3d(LEN, -LEN, LEN);
	glVertex3d(LEN, -LEN, -LEN);

	glNormal3d(0.0, 0.0, -0.0);
	glVertex3d(-LEN, -LEN, -LEN);
	glVertex3d(-LEN, LEN, -LEN);
	glVertex3d(LEN, LEN, -LEN);
	glVertex3d(LEN, -LEN, -LEN);

	glNormal3d(-1.0, 0.0, 0.0);
	glVertex3d(-LEN, -LEN, -LEN);
	glVertex3d(-LEN, -LEN, LEN);
	glVertex3d(-LEN, LEN, LEN);
	glVertex3d(-LEN, LEN, -LEN);

	glNormal3d(0.0, 1.0, 0.0);
	glVertex3d(-LEN, LEN, -LEN);
	glVertex3d(-LEN, LEN, LEN);
	glVertex3d(LEN, LEN, LEN);
	glVertex3d(LEN, LEN, -LEN);

	glNormal3d(0.0, 0.0, 1.0);
	glVertex3d(-LEN, -LEN, LEN);
	glVertex3d(-LEN, LEN, LEN);
	glVertex3d(LEN, LEN, LEN);
	glVertex3d(LEN, -LEN, LEN);

	glNormal3d(1.0, 0.0, 0.0);
	glVertex3d(LEN, -LEN, -LEN);
	glVertex3d(LEN, -LEN, LEN);
	glVertex3d(LEN, LEN, LEN);
	glVertex3d(LEN, LEN, -LEN);
	glEnd();
	glPopMatrix();
}