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
 * @author Mark Hall (mhall{[at]}waikato{[dot]}ac{[dot]}nz)
 * @version $Revision: $
 */
public class DL4jScheme {

  /**
   * A list of those global schemes that are not supported in DL4j
   */
  protected static List<String>
      s_excludedSchemes =
      Arrays.asList( "Naive Bayes", "Naive Bayes incremental", "Naive Bayes multinomial", "Decision tree classifier",
          "Decision tree regressor", "Random forest classifier", "Random forest regressor", "Gradient boosted trees",
          "Support vector regressor", "Multi-layer perceptron classifier", "Multi-layer perceptron regressor",
          "Extreme gradient boosting classifier", "Extreme gradient boosting regressor");

  /**
   * Static factory method for obtaining {@code Scheme} objects encapsulating particular DL4j implementations
   *
   * @param schemeName the name of the scheme to get
   * @return a {@code Scheme} object
   * @throws Exception if a problem occurs
   */
  public static Scheme getSupervisedDlL4jScheme( String schemeName ) throws Exception {
    if ( SupervisedScheme.s_defaultClassifierSchemeList.contains( schemeName ) && !s_excludedSchemes
        .contains( schemeName ) ) {
      return new DL4jClassifierScheme( schemeName );
    } else {
      // TODO do not support unsupervised schemes yet
    }
    throw new UnsupportedSchemeException( "The DL4j engine does not support the " + schemeName + " scheme." );
  }

}
