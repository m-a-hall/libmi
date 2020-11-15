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

package org.phalanxdev.mi;

/**
 * Exception to be thrown when an engine is specified that is not currently supported by PMI
 *
 * @author Mark Hall (mhall{[at]}waikato{[dot]}ac{[dot]}nz)
 * @version $Revision: $
 */
public class UnsupportedEngineException extends Exception {

  /**
   * Constructor without a message
   */
  public UnsupportedEngineException() {
    super();
  }

  /**
   * Constructor with a message
   *
   * @param message the message for the exception
   */
  public UnsupportedEngineException( String message ) {
    super( message );
  }

  /**
   * Constructor with message and cause
   *
   * @param message the message for the exception
   * @param cause   the root cause Throwable
   */
  public UnsupportedEngineException( String message, Throwable cause ) {
    this( message );
    initCause( cause );
    fillInStackTrace();
  }

  /**
   * Constructor with cause argument
   *
   * @param cause the root cause Throwable
   */
  public UnsupportedEngineException( Throwable cause ) {
    this( cause.getMessage(), cause );
  }
}
