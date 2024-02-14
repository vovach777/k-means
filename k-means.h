/*******************************************************************************
 * ALGORITHM IMPLEMENTAIONS
 *
 *  /\  |  _   _  ._ o _|_ |_  ._ _   _
 * /--\ | (_| (_) |  |  |_ | | | | | _>
 *         _|
 *
 * K-MEANS:
 * 	http://en.wikipedia.org/wiki/K-means_clustering
 *
 * First Contributor:
 * 	https://github.com/wycg1984
 ******************************************************************************/

#ifndef ALGO_KMEANS_H__
#define ALGO_KMEANS_H__
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <assert.h>
#include <string.h>
#include <limits>
using namespace std;

namespace alg {
   template <typename T, typename A>
	class KMeans {
		public:
			enum InitMode {
				InitRandom,
				InitManual,
				InitUniform,
			};

			KMeans(int dimNum = 1, int clusterNum = 1) {
				m_dimNum = dimNum;
				m_clusterNum = clusterNum;

				m_means = new T*[m_clusterNum];
				for(int i = 0; i < m_clusterNum; i++) {
					m_means[i] = new T[m_dimNum];
					memset(m_means[i], 0, sizeof(T) * m_dimNum);
				}

				m_initMode = InitRandom;
				m_maxIterNum = 100;
				m_endError = 0.001;
			}

			~KMeans() {
				for(int i = 0; i < m_clusterNum; i++)
				{
					delete[] m_means[i];
				}
				delete[] m_means;
			}

			void SetMean(int i, const T* u) {
				memcpy(m_means[i], u, sizeof(T) * m_dimNum);
			}

			void SetInitMode(int i)	{
				m_initMode = i;
			}

			void SetMaxIterNum(int i) {
				m_maxIterNum = i;
			}

			void SetEndError(T f)	{
				m_endError = f;
			}

			T* GetMean(int i)	{
				return m_means[i];
			}

			int GetInitMode() {
				return m_initMode;
			}

			int GetMaxIterNum()	{
				return m_maxIterNum;
			}

			double GetEndError() {
				return m_endError;
			}


			void Init(const T *data, int N) {
				int size = N;

            if(m_initMode == InitManual)
               return; 					// Do nothing


            auto sample = new T[m_dimNum];

				if(m_initMode ==  InitRandom) {
					int inteval = size / m_clusterNum;

				  // Seed the random-number generator with current time
					srand((unsigned)time(NULL));

					for(int i = 0; i < m_clusterNum; i++) {
						int select = inteval * i + (inteval - 1) * rand() / RAND_MAX;
						for(int j = 0; j < m_dimNum; j++)
							sample[j] = data[select*m_dimNum+j];
						memcpy(m_means[i], sample, sizeof(T) * m_dimNum);
					}

					delete[] sample;
				} else {

					for(int i = 0; i < m_clusterNum; i++) {
						int select = i * size / m_clusterNum;
						for(int j = 0; j < m_dimNum; j++)
							sample[j] = data[select*m_dimNum+j];
						memcpy(m_means[i], sample, sizeof(T) * m_dimNum);
					}

					delete[] sample;
				}
			}

         T clipT(A a) {
            if (a < numeric_limits<T>::min() )
               return numeric_limits<T>::min();
            if (a > numeric_limits<T>::max() )
               return numeric_limits<T>::max();
            return static_cast<T>(a);
         }

			void Cluster(const T *data, int N, int *Label) {
				int size = N;

				assert(size >= m_clusterNum);

				// Initialize model
				Init(data,N);

				// Recursion
				auto x = new T[m_dimNum];	// Sample data
				int label = -1;		// Class index
				int iterNum = 0;
				auto lastCost = 0.0;
				auto currCost = 0.0;
				int unchanged = 0;
				bool loop = true;
				auto counts = new uint32_t[m_clusterNum];
				auto next_means = new A*[m_clusterNum];
				// New model for reestimation
				for(int i = 0; i < m_clusterNum; i++) {
					next_means[i] = new A[m_dimNum];
				}

				while(loop) {
					//clean buffer for classification
					memset(counts, 0, sizeof(uint32_t) * m_clusterNum);
					for(int i = 0; i < m_clusterNum; i++)
					{
						memset(next_means[i], 0, sizeof(A) * m_dimNum);
					}

					lastCost = currCost;
					currCost = 0;

					// Classification
					for(int i = 0; i < size; i++) {
						for(int j = 0; j < m_dimNum; j++)
							x[j] = data[i*m_dimNum+j];

						currCost += GetLabel(x, &label);

						counts[label]++;
						for(int d = 0; d < m_dimNum; d++)
						{
							next_means[label][d] += x[d];
						}
					}
					currCost /= size;

					// Reestimation
					for(int i = 0; i < m_clusterNum; i++) {
						if(counts[i] > 0) {
							for(int d = 0; d < m_dimNum; d++) {

                        next_means[i][d] /= counts[i];
							   m_means[i][d] = clipT( next_means[i][d] );
                        //clipT( next_means[i][d] /= counts[i] );

							}
							//memcpy(m_means[i], next_means[i], sizeof(T) * m_dimNum);
                     // for(int d = 0; d < m_dimNum; d++) {
                     //    m_means[i][d] = next_means[i][d];
                     // }
						}
					}

					// Terminal conditions
					iterNum++;
					if(fabs(lastCost - currCost) < m_endError * lastCost) {
						unchanged++;
					}

					if(iterNum >= m_maxIterNum || unchanged >= 3)
					{
						loop = false;
					}
				}

				// Output the label file
            if (Label) {
               for(int i = 0; i < size; i++) {
                  for(int j = 0; j < m_dimNum; j++)
                     x[j] = data[i*m_dimNum+j];
                  GetLabel(x,&label);
                  Label[i] = label;
               }
            }
				delete[] counts;
				delete[] x;
				for(int i = 0; i < m_clusterNum; i++) {
					delete[] next_means[i];
				}
				delete[] next_means;
			}

		private:
			int m_dimNum;
			int m_clusterNum;
			T** m_means;

			int m_initMode;
			int m_maxIterNum;
			double m_endError;
			double GetLabel(const T* sample, int* label) {
				double dist = -1;
				for(int i = 0; i < m_clusterNum; i++) {
					double temp = CalcDistance(sample, m_means[i]);
					if(temp < dist || dist == -1) {
						dist = temp;
						*label = i;
					}
				}
				return dist;
			}

			double CalcDistance(const T* x,const T* u) {

				double temp = 0;
				for(int d = 0; d < m_dimNum; d++) {
					temp += (x[d] - u[d]) * (x[d] - u[d]);
				}
				return sqrt(temp);
			}
	};
}
#endif
