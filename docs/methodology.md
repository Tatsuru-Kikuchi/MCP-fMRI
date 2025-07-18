# MCP-fMRI Methodology

## Study Design

### Participants
- **Sample Size**: 156 Japanese adults (78 Female, 78 Male)
- **Age Range**: 21-27 years (Mean: 24.7 ± 4.2)
- **Recruitment**: University students and community volunteers in Tokyo, Japan
- **Exclusion Criteria**: 
  - History of neurological disorders
  - Current psychiatric medication
  - MRI contraindications
  - Non-native Japanese speakers

### fMRI Data Acquisition

#### Scanner Specifications
- **Scanner**: 3T Siemens Magnetom Prisma
- **Head Coil**: 64-channel phased array
- **Functional Sequence**: 
  - TR = 2000ms
  - TE = 30ms
  - Flip angle = 90°
  - Voxel size = 3×3×3mm
  - 40 axial slices

#### Mathematical Tasks
1. **Arithmetic Operations**: Addition, subtraction, multiplication, division
2. **Algebraic Problem Solving**: Linear equations, quadratic functions
3. **Geometric Reasoning**: Spatial transformations, area calculations
4. **Statistical Concepts**: Probability, mean, variance

### Data Preprocessing

#### Standard Pipeline
1. **Motion Correction**: SPM12 realignment
2. **Slice Timing Correction**: Temporal interpolation
3. **Spatial Normalization**: MNI152 template
4. **Smoothing**: 8mm FWHM Gaussian kernel
5. **Temporal Filtering**: High-pass filter (128s)

#### Quality Control
- Motion parameters < 3mm translation, < 3° rotation
- Signal-to-noise ratio > 100
- Temporal signal-to-noise ratio > 50

### Statistical Analysis

#### Primary Measures
- **Gender Similarity Index**: 1 - |Cohen's d|
- **Neural Pattern Similarity**: Multi-voxel pattern analysis
- **Performance Overlap**: Distribution overlap coefficient
- **Effect Sizes**: Cohen's d with 95% confidence intervals

#### Brain Regions of Interest
1. **Prefrontal Cortex**: Executive control, working memory
2. **Parietal Lobe**: Numerical processing, spatial attention
3. **Temporal Lobe**: Language processing, semantic memory
4. **Occipital Lobe**: Visual processing
5. **Cerebellum**: Motor control, cognitive functions
6. **Subcortical**: Basal ganglia, thalamus

### Cultural Assessment

#### Japanese Cultural Factors
- **Collectivist Education Score**: Group learning orientation
- **Social Harmony Index**: Conflict avoidance tendencies
- **Family Expectations**: Parental academic pressure
- **Group Achievement**: Collective success orientation
- **Traditional Gender Roles**: Adherence to conventional expectations

#### Measurement Tools
- Cultural Values Scale (Japanese version)
- Educational Environment Questionnaire
- Family Dynamics Inventory
- Gender Role Attitudes Scale

### Machine Learning Analysis

#### Classification Approach
- **Algorithm**: Random Forest (n_estimators=100)
- **Features**: Brain activation patterns (6 regions × multiple voxels)
- **Target**: Gender classification (binary)
- **Validation**: 10-fold cross-validation
- **Performance Metrics**: Accuracy, precision, recall, F1-score

#### Feature Engineering
- **Standardization**: Z-score normalization
- **Dimensionality Reduction**: Principal Component Analysis (95% variance)
- **Feature Selection**: Recursive feature elimination

### Statistical Power Analysis

#### Sample Size Justification
- **Expected Effect Size**: d = 0.1 (small effect)
- **Statistical Power**: 0.80
- **Alpha Level**: 0.05 (two-tailed)
- **Required Sample Size**: 156 participants (78 per group)

### Ethical Considerations

#### IRB Approval
- Institutional Review Board approval obtained
- Written informed consent from all participants
- Right to withdraw without penalty
- Data anonymization and secure storage

#### Cultural Sensitivity
- Japanese research team leadership
- Culturally appropriate recruitment materials
- Native language administration
- Respect for cultural values and practices

### Data Analysis Pipeline

#### Quality Assurance
1. **Data Validation**: Range checks, outlier detection
2. **Missing Data**: Multiple imputation for <5% missing
3. **Assumption Testing**: Normality, homoscedasticity
4. **Robust Statistics**: Bootstrap confidence intervals

#### Reproducibility
- **Open Data**: Anonymized dataset available
- **Code Sharing**: All analysis scripts provided
- **Version Control**: Git repository with full history
- **Computational Environment**: Docker container specifications

### Limitations

#### Study Limitations
- **Cross-sectional Design**: Cannot infer causality
- **University Sample**: Limited generalizability
- **Task Specificity**: Mathematical cognition focus
- **Cultural Specificity**: Japanese population only

#### Technical Limitations
- **Spatial Resolution**: 3mm voxel size
- **Temporal Resolution**: 2-second TR
- **Scanner Differences**: Single scanner site
- **Preprocessing Choices**: Standard pipeline assumptions

### Future Directions

#### Methodological Improvements
- **Longitudinal Design**: Track development over time
- **Multi-site Study**: Cross-cultural validation
- **Higher Resolution**: 7T fMRI, advanced sequences
- **Behavioral Measures**: Real-time performance tracking

#### Research Extensions
- **Other Cognitive Domains**: Language, spatial reasoning
- **Developmental Perspectives**: Children and adolescents
- **Individual Differences**: Personality, motivation factors
- **Intervention Studies**: Educational approach effectiveness