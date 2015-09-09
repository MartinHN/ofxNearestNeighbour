/*
 *  NearestNeighbour.h
 *
 *  Copyright (c) 2013, Neil Mendoza, http://www.neilmendoza.com
 *  All rights reserved. 
 *  
 *  Redistribution and use in source and binary forms, with or without 
 *  modification, are permitted provided that the following conditions are met: 
 *  
 *  * Redistributions of source code must retain the above copyright notice, 
 *    this list of conditions and the following disclaimer. 
 *  * Redistributions in binary form must reproduce the above copyright 
 *    notice, this list of conditions and the following disclaimer in the 
 *    documentation and/or other materials provided with the distribution. 
 *  * Neither the name of Neil Mendoza nor the names of its contributors may be used 
 *    to endorse or promote products derived from this software without 
 *    specific prior written permission. 
 *  
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 *  POSSIBILITY OF SUCH DAMAGE. 
 *
 */
#pragma once

#include "nanoflann.hpp"
#include "PointCloud.h"

namespace itg
{
    using namespace nanoflann;
    
    // T is type of nn point
    template<class T>
    class NearestNeighbour
    {
    public:
        typedef KDTreeSingleIndexAdaptor<
        L2_Simple_Adaptor<float, PointCloud<T> > ,
        PointCloud<T>,
        T::DIM> KdTree;
        
        typedef  typename L1_Adaptor<float, PointCloud<T> >::DistanceType DistanceType;
        
        NearestNeighbour() : kdTree(T::DIM, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */))
        {
            hasBeenBuilt = false;
        }
        void clear(){
            hasBeenBuilt = false;
        }
        
        void buildIndex(const vector<T>& points)
        {
            cloud.points = points;
            if (points.empty()) ofLogError() << "Cannot build index with no points.";
            else {kdTree.buildIndex();
                hasBeenBuilt=true;}
        }
        
        void findNClosestPoints(const T& point, unsigned n, vector<size_t>& indices, vector<float>& distsSquared)
        {
            if(hasBeenBuilt){
            indices.resize(n);
            distsSquared.resize(n);
            kdTree.knnSearch(point.getPtr(), n, &indices[0], &distsSquared[0]);
            }
        }
        
        unsigned findPointsWithinRadius(const T& point, float radius, vector<pair<size_t, float> >& matches)
        {
                        if(hasBeenBuilt){
            nanoflann::SearchParams params;
            return kdTree.radiusSearch(point.getPtr(), radius * radius, matches, params);
                        }
        }
        
        void dbscan(vector<int> *classes, int k, double eps,int minNum)
        {
            // adapted from https://github.com/arpesenti/peopleTracker/blob/master/dbscanClustering.cpp
            if(hasBeenBuilt){
                kdTree.buildIndex();
                size_t dim = T::DIM;
                size_t numPoints = cloud.points.size();
                int C = 0; /* class id assigned to the points in the same cluster */
                int reserveSize = 50; /* allocation dimension for efficiency reasons */
                
                uint8_t *visited = (uint8_t *)calloc(numPoints,sizeof(uint8_t));
                
                const double search_radius = static_cast<double>(eps*eps);
                nanoflann::SearchParams params;
                params.sorted = false;
                
                T queryPoint;
                
                queue<int> neigh;
                std::vector<std::pair<size_t, DistanceType> >  ret_matches;
                vector<int > backupV(minNum,0);
                
                int matches = 0;
                int n=0;
                // start clusterization
                for(int i=0; i<numPoints; i++) {
                    if(visited[i] == 0) {
                        ret_matches.clear();
                        ret_matches.reserve(reserveSize);
                        queryPoint = cloud.points[i];                    // find the number of neighbors of current processed point
                        const size_t nMatches = kdTree.radiusSearch(&queryPoint[0], search_radius, ret_matches, params);
                        visited[i] = 1;
                        n++;
                        if(nMatches <k ||
                           ( std::is_same<T,ofVec3f>::value && ((ofVec3f)queryPoint).lengthSquared()>.26 ) )
                        {
                            // outlier
                            
                            classes->at(i) = 0;
                            
                        } else  {
                            // core point - start expanding a new cluster
                            
                            
                            C = C+1; /* class id of the new cluster */
                            
                            
                            classes->at(i) = C;
                            
                            matches = 1;
                            backupV[matches-1] = i;
                            for(int j=0;j<nMatches;j++) {
                                size_t idx = ret_matches[j].first;
                                if(visited[idx] == 0 && idx!=i) {
                                    neigh.push(idx); /* insert neighbors in the neighbors list */
                                    visited[idx] = 1;
                                    n++;
                                }
                            }
                            // expand cluster until the neighbors list becomes empty
                            while(!neigh.empty()) {
                                int id = neigh.front();
                                
                                neigh.pop();
                                queryPoint = cloud.points[id];
                                
                                ret_matches.clear();
                                ret_matches.reserve(reserveSize);
                                
                                // find the number of neighbors of current processed neighbor point
                                const size_t nMatches = kdTree.radiusSearch(&queryPoint[0], search_radius, ret_matches, params);
                                
                                if (nMatches <= k) {
                                    // border point
                                    classes->at(id) = -C;
                                    matches++;
                                    if(matches < minNum)backupV[matches-1] = id;
                                    
                                } else {
                                    
                                    for(int j=0;j<nMatches;j++) {
                                        size_t idx = ret_matches[j].first;
                                        if(visited[idx] == 0 && idx!=id) {
                                            neigh.push(idx); /* insert neighbors in the neighbors list */
                                            visited[idx] = 1;
                                            n++;
                                        }
                                    }
                                }
                                
                                if(classes->at(id) == 0){
                                    classes->at(id) = C;
                                    matches++;
                                    if(matches < minNum)backupV[matches-1] = id;
                                }
                                
                            }
                            
                            if(matches < minNum){
                                for(int j = 0 ; j < matches ; j++){
                                    classes->at(backupV[j]) = 0;
                                }
                            }
                            
                            cout<< C<< " : " << matches <<  endl;
                        }
                        
                    }
                }
                cout << "visited : " << n << endl;
                cout << "total : " << numPoints<< endl;
                
                free(visited);
            }
        }
        
    
        
        
        bool hasBeenBuilt = false;

        
    private:
        KdTree kdTree;
        PointCloud<T> cloud;
    };
}
