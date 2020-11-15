/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package org.phalanxdev.mi.engines;

import org.phalanxdev.mi.SchemeUtils;
import org.phalanxdev.mi.SupervisedScheme;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.MultiClassClassifier;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.WekaPackageClassLoaderManager;

import java.util.List;
import java.util.Map;

/**
 * Implements classification and regression schemes based on the Spark MLlib library
 *
 * @author Mark Hall (mhall{[at]}waikato{[dot]}ac{[dot]}nz)
 * @version $Revision: $
 */
public class MLlibClassifierScheme extends SupervisedScheme {

  /**
   * Holds the underlying classification or regression scheme. This will be the wrapper classifiers implemented in
   * Weka's distributedWekaSparkDev package
   */
  protected Classifier m_underlyingScheme;

  /**
   * Constructor
   *
   * @param schemeName the name of the scheme to instantiate
   * @throws Exception if a problem occurs
   */
  public MLlibClassifierScheme( String schemeName ) throws Exception {
    super( schemeName );

    if ( schemeName.equalsIgnoreCase( "Logistic regression" ) ) {
      instantiateMLlibClassifier( "MLlibLogistic" );
    } else if ( schemeName.equalsIgnoreCase( "Naive Bayes" ) ) {
      instantiateMLlibClassifier( "MLlibNaiveBayes" );
    } else if ( schemeName.equalsIgnoreCase( "Naive Bayes multinomial" ) ) {
      instantiateMLlibClassifier( "MLlibNaiveBayes" );
      setMLlibClassifierOptions( "-model-type multinomial" );
    } else if ( schemeName.equalsIgnoreCase( "Decision tree classifier" ) ) {
      instantiateMLlibClassifier( "MLlibDecisionTree" );
    } else if ( schemeName.equalsIgnoreCase( "Decision tree regressor" ) ) {
      instantiateMLlibClassifier( "MLlibDecisionTree" );
      setMLlibClassifierOptions( "-impurity variance" );
    } else if ( schemeName.equalsIgnoreCase( "Linear regression" ) ) {
      instantiateMLlibClassifier( "MLlibLinearRegressionSGD" );
    } else if ( schemeName.equalsIgnoreCase( "Support vector classifier" ) ) {
      instantiateMLlibClassifier( "MLlibSVM" );
    } else if ( schemeName.equalsIgnoreCase( "Support vector regressor" ) ) {
      throw new Exception( "MLlib engine does not support support vector regression" );
    } else if ( schemeName.equalsIgnoreCase( "Random forest classifier" ) ) {
      instantiateMLlibClassifier( "MLlibRandomForest" );
    } else if ( schemeName.equalsIgnoreCase( "Random forest regressor" ) ) {
      instantiateMLlibClassifier( "MLlibRandomForest" );
      setMLlibClassifierOptions( "-impurity variance" );
    } else if ( schemeName.equalsIgnoreCase( "Gradient boosted trees" ) ) {
      // MLlib's implementation can handle regression via setting the impurity metric
      // for the base trees to variance
      instantiateMLlibClassifier( "MLlibGradientBoostedTrees" );
    }
  }

  /**
   * Checks whether the scheme can handle the data that will be coming in
   *
   * @param data     the header of the data that will be used for training
   * @param messages a list to store messages describing any problems/warnings for the selected scheme with respect to the incoming data
   * @return true if the scheme can handle the data
   */
  @Override public boolean canHandleData( Instances data, List<String> messages ) {
    try {
      Classifier finalClassifier = (Classifier) getConfiguredScheme( data );
      finalClassifier.getCapabilities().testWithFail( data );
    } catch ( Exception ex ) {
      messages.add( ex.getMessage() );
      return false;
    }

    addStringAttributeWarningMessageIfNeccessary( data, messages );

    return false;
  }

  /**
   * Instantiates the Weka MLlib wrapper classifier corresponding to the supplied scheme name
   *
   * @param schemeName the name of the scheme to instantiate
   * @throws Exception if a problem occurs
   */
  protected void instantiateMLlibClassifier( String schemeName ) throws Exception {
    m_underlyingScheme =
        (Classifier) WekaPackageClassLoaderManager.objectForName( "weka.classifiers.mllib." + schemeName );
  }

  /**
   * Set options for the wrapper classifier
   *
   * @param options an option string
   * @throws Exception if a problem occurs
   */
  protected void setMLlibClassifierOptions( String options ) throws Exception {
    ( (OptionHandler) m_underlyingScheme ).setOptions( Utils.splitOptions( options ) );
  }

  /**
   * No MLlib schemes support incremental (row by row) training
   *
   * @return false
   */
  @Override public boolean supportsIncrementalTraining() {
    return false;
  }

  /**
   * MLlib schemes do not support resumable iterative training
   *
   * @return false
   */
  @Override public boolean supportsResumableTraining() {
    return false;
  }

  /**
   * Returns true if the configured scheme can directly handle string attributes
   *
   * @return true if the configured scheme can directly handle string attributes
   */
  @Override public boolean canHandleStringAttributes() {
    return false;
  }

  /**
   * Get a map of meta information about the scheme and its parameters, useful for building editor dialogs
   *
   * @return a map of meta information about the scheme and its parameters
   * @throws Exception if a problem occurs
   */
  @Override public Map<String, Object> getSchemeInfo() throws Exception {
    return SchemeUtils.getSchemeParameters( m_underlyingScheme );
  }

  /**
   * Configure the underlying scheme using the supplied command-line option settings
   *
   * @param options an array of command-line option settings
   * @throws Exception if a problem occurs
   */
  @Override public void setSchemeOptions( String[] options ) throws Exception {
    if ( m_underlyingScheme != null ) {
      ( (OptionHandler) m_underlyingScheme ).setOptions( options );
    }
  }

  /**
   * Get the underlying scheme's command line option settings. This may be different from those
   * that could be obtained from scheme returned by {@code getConfiguredScheme()}, as the configured
   * scheme might be a wrapper (meta classifier) around the underlying scheme.
   *
   * @return the options of the underlying scheme
   */
  @Override public String[] getSchemeOptions() {
    if ( m_underlyingScheme != null ) {
      return ( (OptionHandler) m_underlyingScheme ).getOptions();
    }
    return null;
  }

  /**
   * Set underlying scheme parameters from a map of parameter values. Note that this will set only primitive parameter
   * types on the scheme. It does not process nested objects. This method is used primarily by the GUI editor dialogs. Use
   * setSchemeOptions() to handle all parameters (including those on nested complex objects).
   *
   * @param schemeParameters a map of scheme parameters to set
   * @throws Exception if a problem occurs.
   */
  @Override public void setSchemeParameters( Map<String, Map<String, Object>> schemeParameters ) throws Exception {
    SchemeUtils.setSchemeParameters( m_underlyingScheme, schemeParameters );
  }

  /**
   * Return the underlying predictive scheme, configured and ready to use. The incoming training data
   * is supplied so that the scheme can decide (based on data characteristics) whether the underlying scheme
   * needs to be combined with data filters in order to be applicable to the data. E.g. The user might have selected
   * logistic regression which, in the given engine, can only support binary class problems. At execution time, the
   * incoming data could have more than two class labels, in which case the underlying scheme will need to be wrapped
   * in a MultiClassClassifier.
   *
   * @param incomingHeader the header of the incoming training data
   * @return the underlying predictive scheme
   * @throws Exception if there is a problem configuring the scheme
   */
  @Override public Object getConfiguredScheme( Instances incomingHeader ) throws Exception {
    Classifier finalScheme = adjustForSamplingAndPreprocessing( incomingHeader, m_underlyingScheme );

    if ( finalScheme.getClass().toString().endsWith( "SVM" ) || finalScheme.getClass().toString()
        .endsWith( "GradientBoostedTrees" ) ) {
      // wrap in a MultiClassClassifier if the class is nominal with more than two values
      if ( incomingHeader.classAttribute().isNominal() && incomingHeader.classAttribute().numValues() > 2 ) {
        MultiClassClassifier temp = new MultiClassClassifier();
        temp.setClassifier( finalScheme );
        finalScheme = temp;
      }
    }

    return finalScheme;
  }

  public void setConfiguredScheme( Object scheme ) throws Exception {
    if ( scheme instanceof FilteredClassifier ) {
      scheme = ( (FilteredClassifier) scheme ).getClassifier();
    }

    if ( !scheme.getClass().getCanonicalName().equals( m_underlyingScheme.getClass().getCanonicalName() ) ) {
      throw new Exception(
          "Supplied configured scheme " + scheme.getClass().getCanonicalName() + " does not " + "match "
              + m_underlyingScheme.getClass().getCanonicalName() );
    }

    // Just copy over option settings from the supplied scheme, so that we avoid consuming
    // memory for large trained models (model gets loaded again when transformation is executed)
    ((OptionHandler) m_underlyingScheme).setOptions( ((OptionHandler) scheme).getOptions() );
    // m_underlyingScheme = (Classifier) scheme;
  }
}
