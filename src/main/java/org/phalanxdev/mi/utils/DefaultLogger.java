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

package org.phalanxdev.mi.utils;

import weka.core.logging.Logger;
import weka.core.logging.Logger.Level;

/**
 * Default log adapter that uses Weka logging.
 *
 * @author Mark Hall (mhall{[at]}waikato{[dot]}ac{[dot]}nz)
 * @version 1: $
 */
public class DefaultLogger implements IMILogAdapter {

  public DefaultLogger() {
    Logger.log(Level.INFO, "Logging started");
  }

  @Override
  public boolean isBasic() {
    return true;
  }

  @Override
  public boolean isDebug() {
    return false;
  }

  @Override
  public boolean isDetailed() {
    return false;
  }

  @Override
  public boolean isError() {
    return false;
  }

  @Override
  public boolean isRowLevel() {
    return false;
  }

  @Override
  public void logBasic(String message) {
    weka.core.logging.Logger.log(Level.INFO, message);
  }

  @Override
  public void logBasic(String message, Object... aux) {
    weka.core.logging.Logger.log(Level.INFO, message);
  }

  @Override
  public void logDetailed(String message) {
    weka.core.logging.Logger.log(Level.FINE, message);
  }

  @Override
  public void logDetailed(String message, Object... aux) {
    weka.core.logging.Logger.log(Level.FINE, message);
  }

  @Override
  public void logMinimal(String message) {
    weka.core.logging.Logger.log(Level.OFF, message);
  }

  @Override
  public void logMinimal(String message, Object... aux) {
    weka.core.logging.Logger.log(Level.OFF, message);
  }

  @Override
  public void logDebug(String message) {
    weka.core.logging.Logger.log(Level.FINEST, message);
  }

  @Override
  public void logDebug(String message, Object... aux) {
    weka.core.logging.Logger.log(Level.FINEST, message);
  }

  @Override
  public void logError(String message) {
    weka.core.logging.Logger.log(Level.SEVERE, message);
  }

  @Override
  public void logError(String message, Object... aux) {
    weka.core.logging.Logger.log(Level.SEVERE, message);
  }
}
