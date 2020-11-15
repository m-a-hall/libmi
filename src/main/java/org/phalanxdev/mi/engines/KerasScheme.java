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

import org.phalanxdev.mi.UnsupportedSchemeException;
import org.phalanxdev.mi.Scheme;
import org.phalanxdev.mi.SupervisedScheme;

import java.util.Arrays;
import java.util.List;

/**
 * Base class for schemes supported in the Keras engine. Only the zoo classifier is supported so far...
 *
 * @author Mark Hall (mhall{[at]}waikato.ac.nz)
 * @version $Revision: $
 */
public abstract class KerasScheme {

  /**
   * A list of those global schemes that are not supported by the Keras engine
   */
  protected static List<String>
      // TODO need to break out zoo classifiers into individual schemes, then can add the generic "Deep learning network" back into this list
      s_excludedSchemes =
      Arrays.asList( "Naive Bayes", "Naive Bayes incremental", "Naive Bayes multinomial", "Decision tree classifier",
          "Decision tree regressor", "Random forest classifier", "Random forest regressor", "Gradient boosted trees",
          "Support vector regressor", "Support vector classifier", "Logistic regression", "Linear regression",
          "Multi-layer perceptron classifier", "Multi-layer perceptron regressor",
          "Extreme gradient boosting classifier", "Extreme gradient boosting regressor",
          "Multi-layer perceptron classifier", "Multi-layer perceptron regressor" );

  /**
   * Returns a Scheme object configured for the Keras engine.
   *
   * @param schemeName the name of the scheme to get
   * @return a configured Scheme object
   * @throws Exception if the scheme is not supported in this engine.
   */
  public static Scheme getSupervisedKerasScheme( String schemeName ) throws Exception {
    if ( SupervisedScheme.s_defaultClassifierSchemeList.contains( schemeName ) && !s_excludedSchemes
        .contains( schemeName ) ) {
      return new KerasClassifierScheme( schemeName );
    } else {
      // TODO do not support unsupervised schemes yet
    }
    throw new UnsupportedSchemeException( " The Keras engine does not support the " + schemeName + " scheme." );
  }
}
