# cs6140_a2

## Reference

Distance metric:
- Normalized Euclidean Distance between data points $x, y$: $D = \sqrt{ \sum{ \frac{(x_i - y_i)^2}{\sigma_i} } }$
    - $i$ = column

Column 0: "Avg. Area Income"
- Min: 17796.6311895434
- Max: 107701.748377639
- Mean: 68583.10898395974
- Median: 68804.28640371619
- Variance: 113592776.71408534
- Standard Deviation: 10657.99121383037

Column 1: "Avg. Area House Age"
- Min: 2.64430418603671
- Max: 9.51908806613594
- Mean: 5.977222035280273
- Median: 5.970428947124805
- Variance: 0.9829853565194727
- Standard Deviation: 0.9914561798281721

Column 2: "Avg. Area Number of Rooms"
- Min: 3.2361940234262
- Max: 10.7595883359386
- Mean: 6.987791850907944
- Median: 7.002901987201935
- Variance: 1.0117004891419086
- Standard Deviation: 1.0058332312773866

Column 3: "Area Population"
- Min: 172.6106862729
- Max: 69621.7133777904
- Mean: 36163.51603857466
- Median: 36199.40668926055
- Variance: 98518530.1756471
- Standard Deviation: 9925.650113501235

Column 4: "Price"
- Min: 15938.6579232878
- Max: 2469065.5941747
- Mean: 1232072.654142357
- Median: 1232669.37796579
- Variance: 124692058202.24155
- Standard Deviation: 353117.6265810609

<b>Price Histogram:</b>

![alt text](./resources/price_hist.png)

## Logistics

Dependency installation (if virtual environment is used):
- `pip install numpy scipy matplotlib scikit-learn pandas`

Jake:

- [ ] Create initial distance metric (Thursday night)
- [ ] Create confusion matrix (1.C)
- [ ] 2.D and 2.E
- [ ] 4
- [ ] Do more exploration in the final task, such as trying different numbers of dimensions after PCA.

Yihan:
- [ ] 1.C, 1.D
- [ ] 2.A-C
- [ ] 3
- [ ] Try out more clustering methods. done
- [ ] Implement additional cluster quality metrics. done
- [ ] Implement confusion matrix

Files:
1) KNN
2) Clustering
3) PCA
4) Main


what's left
- report -jake,yihan
- readMe - jake,yihan
- try different K - yihan(done)
- modify price range to get better precision - jake, yihan (done)
- modify distance metrics function to a better runtime and make sure it is correct -jake, yihan (done)
- 2.E experience different eigenvector weights -jake
- try different classfiers on task for4
- adding comments
