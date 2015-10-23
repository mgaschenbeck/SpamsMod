#include <iostream>
#include <string>
#include <sstream>
#include "E:\GitClones\digitalglobeSPAMS\digitalglobeSPAMS\digitalglobeSPAMS\inc\dictLearn\dicts.h"

using namespace std;




int main(int argc, char *argv[])
{
	//Set some parameters.  The following line is the python parameters we've used in the past
	//paramCloud = { 'modeParam' : 3, 'K' : dictionarySize, 'lambda1' : .15, 'numThreads' : -1, 'iter' : 750 }#K = 250
	ParamDictLearn<float> parameters;
	parameters.modeParam = PARAM3;
	parameters.p = 250;
	parameters.lambda = .15;
	parameters.iter = 750;

	//Here is the actual training data in matrix form 64xN rows x cols
	string cloudPatches = "E:\\Python\\cloudPatches_patchSize_8_resolution_16_skipFactor_25.txt";


	//First, grab the first line to see how many samples we have.  Put the samples in our matrix and keep track of the number
	ifstream instream(cloudPatches);
	string line,token;
	stringstream s0;
	int numSamples = 0;

	getline(instream, line);
	s0<<line;
	while (!s0.eof())
	{
		s0 >> token;
		numSamples++;
	}
	cout << "numSamples = " << numSamples << endl;


	//Now, we can set the trainingdata array size, and do the rest of the lines
	float *trainingdata = new float[64 * numSamples];
	//NOTE:  Training data is unpacked column wise!
	//i.e. trainingdata[1]=trainingdata[1,0], trainingdata[64]=trainingdata[0,1] for 64 dim data
	
	for (int i = 0; i < 64; i++)
	{	
		//line is already set from above, redoing the first line
		stringstream ss(line);
		for (int j = 0; j < numSamples; j++)
		{
			ss >> trainingdata[j*64 + i];
		}

		instream >> line;
	}
	
	//Now we setup the Matrix object that is used by spams
	Matrix<float> trainingData(trainingdata, 64, 250);
	//cout << "i=0,1   " << trainingData(0,1) << endl;
	//cout << "i=1,0   " << trainingData(1,0) << endl;
	
	//Now do the actual training
	Trainer<float> test(250);
	test.train(trainingData, parameters);

	//Write the dictionary to a text file
	ofstream newOut("E:/Python/testOutCloudDict.txt");
	for (int i = 0; i < 64; i++)
	{
		for (int j = 0; j < numSamples; j++)
		{
			newOut << trainingData(i, j) << ",";
		}
		newOut << endl;
	}
	newOut.close();

	return -1;
}

