# Repository for Notes & To-Do List

### Features to Consider Adding
- NLP on 'features' column
    - e.g. if listing has a feature that is all caps, all lower-case, excessive punctuation (\*\*\*Great Deal\*\*\*), etc.



### Look Into
- Most predictive features in 'features' column, as defined by:
    - Biggest gap between high and med/low interest
    - Most frequently occurring
- Distribution of manager_id, and mean interest for most frequent manager id's
    - it may be that certain managers do a poor job of maintaining buildings so interest is lower, or vice versa
- Distribution of display address by interest level


### Thoughts on Modeling
- Consider neural network to learn to classify associated photos by interest level
- Not too many outliers, but I suspect they'll all be med/low interest based on investigations so far; may want to do an ensemble where outliers get treated separately
- Given the density of data, and small bits of signal distributed across many features, a naive bayes classifier might work well
