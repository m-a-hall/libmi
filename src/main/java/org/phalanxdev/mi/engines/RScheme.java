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
 * Implementation of a {@code Scheme} for the MLR R engine.
 *
 * @author Mark Hall (mhall{[at]}waikato.ac.nz)
 * @version $Revision: $
 */
public abstract class RScheme {

  /**
   * A list of those schemes that are not available in MLR
   */
  protected static List<String>
      s_excludedSchemes =
      Arrays.asList( "Naive Bayes incremental", "Naive Bayes multinomial", /* "Multi-layer perceptron classifier",
          "Multi-layer perceptron regressor" ,*/ "Deep learning network", "Extreme gradient boosting classifier",
          "Extreme gradient boosting regressor" );

  /**
   * Static factory method for obtaining a {@code Scheme} instance that encapsulates an MRL R implementation of the
   * named scheme
   *
   * @param schemeName the name of the scheme to get
   * @return a {@code Scheme} object
   * @throws Exception if a problem occurs
   */
  protected static Scheme getSupervisedRScheme( String schemeName ) throws Exception {
    if ( SupervisedScheme.s_defaultClassifierSchemeList.contains( schemeName ) && !s_excludedSchemes
        .contains( schemeName ) ) {
      return new RClassifierScheme( schemeName );
    } else {
      // TODO clusterers?
    }
    throw new UnsupportedSchemeException( "The R engine (MLR) does not support the " + schemeName + " scheme." );
  }
}
