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

import org.phalanxdev.mi.EngineNotAvailableException;
import org.phalanxdev.mi.PMIEngine;
import org.phalanxdev.mi.UnsupportedSchemeException;
import org.phalanxdev.mi.Scheme;
import org.phalanxdev.mi.SupervisedScheme;
import weka.core.WekaPackageClassLoaderManager;

import java.util.List;

/**
 * @author Mark Hall (mhall{[at]}waikato.ac.nz)
 * @version $Revision: $
 */
public class DL4jEngine extends PMIEngine {

  /**
   * Name of the engine
   */
  public static final String ENGINE_NAME = "DL4j";

  public static final String ENGINE_CLASS = DL4jEngine.class.getCanonicalName();

  @Override public String engineName() {
    return ENGINE_NAME;
  }

  @Override public boolean engineAvailable( List<String> messages ) {

    try {
      WekaPackageClassLoaderManager.forName( "weka.classifiers.functions.Dl4jMlpClassifier" );
      return true;
    } catch ( ClassNotFoundException e ) {
      if ( messages != null ) {
        messages.add( e.getMessage() );
      }
    }

    return false;
  }

  @Override public boolean supportsScheme( String schemeName ) {
    return SupervisedScheme.s_defaultClassifierSchemeList.contains( schemeName ) && !DL4jScheme.s_excludedSchemes
        .contains( schemeName );
  }

  @Override public Scheme getScheme( String schemeName )
      throws EngineNotAvailableException, UnsupportedSchemeException {

    if ( !engineAvailable( null ) ) {
      throw new EngineNotAvailableException( engineName() + " is not available!" );
    }

    try {
      return DL4jScheme.getSupervisedDlL4jScheme( schemeName );
    } catch ( Exception ex ) {
      throw new UnsupportedSchemeException( ex );
    }
  }
}
