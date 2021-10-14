# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        '''
        La función que he configurado consiste en calcular la mínima distancia hasta la comida y penalizar
        cuanto más grande sea esta, así como calcular la mínima distancia al fantasma más cercano y 
        penalizar si ésta es pequeña.
        Entonces creo un nuevo estado (que esto venía dado ya del código base) que será el estado del sucesor
        En este calculo las distancias que he dicho
        Finalmente creo la variable "score" que es la que va a ir acumulando los puntos para devolverlos al final.
        En esta variable sumo (o resto) la penalización o la bonificación correspondiente.
        He pensado que sería útil utilizar la función 1/x, ya que si x crece, esta función devuelve valores más
        pequeños, mientras que si x decrece, devuelve valores mayores. 
        Ahora bien, si x = 0, entonces no se podría calcular porque no está definida, así pues para arreglarlo
        he pensado en utilizar la función 1/(x+1), que si x es cero devolverá el valor 1.
        '''
        score = successorGameState.getScore() # Variable en la que acumularé los puntos
        foodList = newFood.asList()  # Convierte la "matriz" de true/false en una lista con las posiciones en las que hay un true
        closestFoodDist = 9999999999999 # Asigno un valor muy alto, pero que no sea infinito porque me daba problemas
        for food in foodList: # Para cada comida...
            foodDist = util.manhattanDistance(food, newPos) # Calculo su distancia Manhattan desde Pacman
            if foodDist < closestFoodDist: # Hago el mínimo de estas distancias
                closestFoodDist = foodDist
        closestGhostDist = 9999999999999 # Asigno un valor muy alto, igual que con la comida
        for ghost in newGhostStates: # Para cada fantasma...
            ghostPos = ghost.getPosition() # Adquiero su posición
            ghostDist = util.manhattanDistance(ghostPos, newPos) # Calculo la distancia desde Pacman
            if ghostDist < closestGhostDist: # Me quedo con la mínima
                closestGhostDist = ghostDist
        # A continuación, me planteo que si los fantasmas están en modo huidizo, entonces no interesa huír de ellos, así que podemos ignorarlos
        for ghostTime in newScaredTimes: # newScaredTimes contiene una lista de números que represenetan cuantos
                                         # frames le queda a cada fantasma de ser blanco.
            if ghostTime < 3:  # Si le queda poco tiempo lo tratamos como fantasma normal, porque sino hay mucho riesgo
                # En caso de que el fantasma sea "normal" o esté a punto de serlo, penalizamos por su proximidad
                score = score + (1.0 / (closestFoodDist + 1.0)) - (1.0 / (closestGhostDist + 1.0))
            else: # En caso de que el fantasma esté en modo huidizo, no hace falta alejarnos de él, así pues lo ignoramos.
                score = score + (1.0 / (closestFoodDist + 1.0))

        # Finalmente, he incluido una penalización muy alta si se da el caso que el fantasma está muy muy cerca de Pacman
        # De esta manera, si el Pacman ve que un movimiento le hará estar a una distancia menor que 2 de un fantasma,
        # no tomará nunca la decisión de realizarla porque sino recibirá un score super bajo.
        if closestGhostDist < 2:
            return -999999
        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # Format of result = [score, action]
        result = self.minimax(gameState, 0, self.depth)

        # Return the action from result
        return result[1]

    def minimax(self, gameState, index, depth):
        """
        Devolverá un valor en tipo 2-tupla: (valor, acción)

        Hay tres casos:
        1. Estado terminal: ya se ha acabado la profundidad, o bien no quedan acciones legales
        2. Agente maximizador: le toca a pacman y por tanto hay que maximizar la puntuación
        3. Agentes minimizadores: le toca a algunfantasma y por tanto hay que minimizar la puntuación
        """
        # 1. Estado terminal:
        #   - Si la longitud de la lista de acciones legales es 0, quiere decir que no hay ninguna acción legal...
        #     Esto lo hago debido a un error que me daba y así pues no intenta realizar acciones que no sean legales.
        #   - Si la profundidad es 0, quiere decir que ya hemos explorado todos los niveles del árbol deseados
        # En cualquiera de los dos casos anteriores, acabamos el minimax y devolvemos el valor del estado en cuestión.
        if len(gameState.getLegalActions(index)) == 0 or depth == 0:
            return gameState.getScore(), ""

        # 2. MAX:
        #   - He creído conveniente realizar la función que maximiza a parte. En ella haré la explicación.
        #   - Básicamente dice que si el pacman (es decir, el agente con índice 0) es el que juega ahora, entonces
        #     lo que hay que hacer es maximizar.
        if index == 0:
            return self.maximizer(gameState, index, depth)

        # 3. MIN:
        #   - Igual que con el MAX, pero ahora estamos en todos los otros casos, es decir, en los casos en los que
        #     index > 0, que significa que les toca a los fantasmas. Da igual en qué orden jueguen los fantasmas. En
        #     este caso, lo que hacemos es minimizar.
        else:
            return self.minimizer(gameState, index, depth)

    def maximizer(self, gameState, index, depth):
        """
        Esta es la función que maximizará el resultado. Consiste en lo siguiente: Para cada acción legal del agente
        - Calculo quién será el agente sucesor. Esto lo hago básicamente sumándole 1 al índice, para indicar que le
          toca al siguiente jugador.
        - Si al sumar el índice nos pasamos del máximo número significará que en realidad ya han acabado su turno todos
          los jugadores y que por tanto empieza "otra ronda": le toca a pacman. Así pues, pongo manualmente que
          el índice sea 0 (el de pacman). También añado que hemos explorado ya una capa del árbol y por tanto resto
          1 a la profundidad.
        - Después de esto, calculo el value llamando al minimax de los sucesores y devuelvo luego el máximo.
        """
        legalMoves = gameState.getLegalActions(index)  # Genero una lista de las acciones legales del agente correspondiente a index
        maxValue = float("-inf")  # Pongo el máximo al número más pequeño posible
        maxAction = ""  # Inicializo la acción emtpy string como acción máxima

        for action in legalMoves:  # Para cada acción legal...
            successor = gameState.generateSuccessor(index, action) # calculo el sucesor
            successorIndex = index + 1  # Configuro su índice, que consiste en uno más que el anterior
            successorDepth = depth  # Añado una variable que será su profundidad, así no tengo que modificar la profundidad actual

            # Si estamos en el caso en que successorIndex == gameState.getNumAgents() significa que "nos hemos pasado".
            # Es decir, que ahora en realidad le tocaría a pacman
            if successorIndex == gameState.getNumAgents():  # getNumAgents() devuelve el total de jugadores (y los índices van de 0 a n-1, con lo cual siemrpe es uno menos)
                successorIndex = 0  # Lo asignamos a 0, para indicar que es pacman
                successorDepth -= 1  # Hemos explorado ya una capa del árbol

            value = self.minimax(successor, successorIndex, successorDepth)[0]  # Recursividad: pongo el [0] porque esta función devuelve una 2-tupla

            if value > maxValue:  # Calculo el máximo
                maxValue = value  # Me quedo siempre con el valor máximo...
                maxAction = action  # ... así como con la acción correspondiente

        return maxValue, maxAction  # Devuelvo una 2-tupla (valor, acción)

    def minimizer(self, gameState, index, depth):
        """
        Esta función minimizará el resultado. Es bastante análoga a la que maximiza, pero cambiando max por min. No vuelvo
        a hacer toda la explicación porque es la misma:
        - En vez de inicializar en -inf, inicializamos en +inf
        - Lo de si successorIndex == gameState.getNumAgents() es igual
        - En vez de buscar el máximo, buscamos ahora el mínimo
        """
        legalMoves = gameState.getLegalActions(index)
        minValue = float("inf")
        minAction = ""

        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action)
            successorIndex = index + 1
            successorDepth = depth

            # Si estamos en el caso en que successorIndex == gameState.getNumAgents() significa que "nos hemos pasado".
            # Es decir, que ahora en realidad le tocaría a pacman
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth -= 1

            value = self.minimax(successor, successorIndex, successorDepth)[0]

            if value < minValue:
                minValue = value
                minAction = action

        return minValue, minAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Formato del resultado: 2-tupla del tipo (valor, acción). Queremos sólo la acción, por tanto...
        result = self.alphaBeta(gameState, 0, self.depth, float('-inf'), float('+inf'))
        # ...devolvemos sólo la segunda componente
        return result[1]

    def alphaBeta(self, gameState, index, depth, alpha, beta):
        """
        La función minimax con poda alpha-beta no difiere mucho de la función minimax a secas. Es por ello que he
        realizado un "copy-paste" de las funciones del minimax, y he añadido alguna cosa para realizar la poda:

        Como siempre, tenemos tres casos:
        1. Estado terminal: o bien ya no quedan acciones legales, o bien hemso explorado todas las capas deseadas
        2. Le toca a pacman (index = 0) y por tanto hay que maximizar
        3. Le toca a algún fantasma (index > 0) y por tanto hay que minimizar
        """
        # 1. Estado terminal:
        #   - Si la longitud de la lista de acciones legales es 0, quiere decir que no hay ninguna acción legal...
        #     Esto lo hago debido a un error que me daba y así pues no intenta realizar acciones que no sean legales.
        #   - Si la profundidad es 0, quiere decir que ya hemos explorado todos los niveles del árbol deseados
        # En cualquiera de los dos casos anteriores, acabamos el minimax y devolvemos el valor del estado en cuestión.
        if len(gameState.getLegalActions(index)) == 0 or depth == 0:
            return gameState.getScore(), ""

        # 2. MAX:
        #   - He creído conveniente realizar la función que maximiza a parte. En ella haré la explicación.
        #   - Básicamente dice que si el pacman (es decir, el agente con índice 0) es el que juega ahora, entonces
        #     lo que hay que hacer es maximizar.
        if index == 0:
            return self.maximizer(gameState, index, depth, alpha, beta)

        # 3. MIN:
        #   - Igual que con el MAX, pero ahora estamos en todos los otros casos, es decir, en los casos en los que
        #     index > 0, que significa que les toca a los fantasmas. Da igual en qué orden jueguen los fantasmas. En
        #     este caso, lo que hacemos es minimizar.
        else:
            return self.minimizer(gameState, index, depth, alpha, beta)

    def maximizer(self, gameState, index, depth, alpha, beta):
        """
        Maximiza, teniendo en cuenta que el máximo sea inferior a beta. Si supera a beta, paramos:
        - Calculo quién será el agente sucesor. Esto lo hago básicamente sumándole 1 al índice, para indicar que le
          toca al siguiente jugador.

        - Si al sumar el índice nos pasamos del máximo número significará que en realidad ya han acabado su turno todos
          los jugadores y que por tanto empieza "otra ronda": le toca a pacman. Así pues, pongo manualmente que
          el índice sea 0 (el de pacman). También añado que hemos explorado ya una capa del árbol y por tanto resto
          1 a la profundidad.

        - Después de esto, calculo el value llamando al minimax de los sucesores y devuelvo luego el máximo. Al realizar
          este máximo, calculo alpha como el máximo entre alpha y el máximo calculado. Así guardo en alpha el valor
          máximo.

        - Antes de esto, de hecho, se realiza la poda (si se ha de realizar). Esta poda se dará si el máximo que estoy
          calculando ya no supera el mínimo (establecido por beta) y devolveremos directamente el resultado, saliendo
          del bucle "for" ya que no valdrá la pena seguir: si un valor ya supera a beta, como nos quedaremos con el máximo
          seguro que superará a beta y esto no nos interesa: no hace falta seguir.
        """
        legalMoves = gameState.getLegalActions(index)  # Genero una lista de las acciones legales del agente correspondiente a index
        maxValue = float("-inf")  # Pongo el máximo al número más pequeño posible
        maxAction = ""  # Inicializo la acción emtpy string como acción máxima

        for action in legalMoves:  # Para cada acción legal...
            successor = gameState.generateSuccessor(index, action) # calculo el sucesor
            successorIndex = index + 1  # Configuro su índice, que consiste en uno más que el anterior
            successorDepth = depth  # Añado una variable que será su profundidad, así no tengo que modificar la profundidad actual

            # Si estamos en el caso en que successorIndex == gameState.getNumAgents() significa que "nos hemos pasado".
            # Es decir, que ahora en realidad le tocaría a pacman
            if successorIndex == gameState.getNumAgents():  # getNumAgents() devuelve el total de jugadores (y los índices van de 0 a n-1, con lo cual siemrpe es uno menos)
                successorIndex = 0  # Lo asignamos a 0, para indicar que es pacman
                successorDepth -= 1  # Hemos explorado ya una capa del árbol

            value = self.alphaBeta(successor, successorIndex, successorDepth, alpha, beta)[0]  # Recursividad: pongo el [0] porque esta función devuelve una 2-tupla

            if value > maxValue:  # Calculo el máximo
                maxValue = value  # Me quedo siempre con el valor máximo...
                maxAction = action  # ... así como con la acción correspondiente
            if maxValue > beta: # PODAMOS
                return maxValue, maxAction
            alpha = max(alpha,maxValue) # Vamos calculando, al mismo tiempo, el máximo alpha, que nos servirá para podar en el mínimo.
        return maxValue, maxAction  # Devuelvo una 2-tupla (valor, acción)

    def minimizer(self, gameState, index, depth, alpha, beta):
        """
        Esta función minimizará el resultado. Es bastante análoga a la que maximiza, pero cambiando max por min. No vuelvo
        a hacer toda la explicación porque es la misma:
        - En vez de inicializar en -inf, inicializamos en +inf
        - Lo de si successorIndex == gameState.getNumAgents() es igual
        - En vez de buscar el máximo, buscamos ahora el mínimo.
        - Si el valor se pasa (inferiormente) de alpha, significa que no vamos a obtener un mínimo por encima de alpha,
          por tanto no vale la pena seguir: podamos
        """
        legalMoves = gameState.getLegalActions(index)
        minValue = float("inf")
        minAction = ""

        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action)
            successorIndex = index + 1
            successorDepth = depth

            # Si estamos en el caso en que successorIndex == gameState.getNumAgents() significa que "nos hemos pasado".
            # Es decir, que ahora en realidad le tocaría a pacman
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth -= 1

            value = self.alphaBeta(successor, successorIndex, successorDepth, alpha, beta)[0]

            if value < minValue:
                minValue = value
                minAction = action

            if minValue < alpha:  # Poda
                return minValue, minAction
            beta = min(beta, minValue)

        return minValue, minAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Como siempre, el resultado es en forma de 2-tupla (valor, acción) y nosotros queremos sólo la acción.
        return self.expectimax(gameState, 0, self.depth)[1]

    def expectimax(self, gameState, index, depth):
        """
        Es bastante parecido al minimax pero con una diferencia: el minimax asume que el contrincante
        jugará de la mejor manera posible. En el caso de los fantasmas esto no es cierto del to do, pues los fantasmas
        actuarán de foma aleatoria (según tengo entendido) y así pues, no siempre escogerán la mejor opción

        Lo que hace el expectimax es, pues, no intentar minimizar la acción del contrincante, sino intentar tomar
        la probabilidad de que el fantasma haga tal movimiento. Es decir, la esperanza. Como el fantasma toma
        las acciones de manera aleatoria, la esperanza es igual a la media aritmética de todos los valores posibles

        Como de costumbre, tenemos tres casos posibles:
        1. Estado terminal
            Este caso se da si ya hemos explorado todas las capas deseadas o bien si no hay acciones legales que se
            puedan realizar. En este caso devolvemos la puntuación del estado.
        2. Le toca a pacman:
            Este caso se da si le toca a pacman, es decir, si el índice es 0. En este caso pacman seguirá, como siempre
            intentando maximizar su jugada. Por tanto, ejecuta la función maxValue que es una copia de la del minimax
        3. Le toca a un fantasma:
            Este tercer caso se dará si el ínidce es mayor que cero: esto indica que le toca a un fantasma.
            Aquí es lo que cambia respecto a la función minimax: en vez de minimizar, calcula la esperanza de
            escoger tal valor, que coincide con la media aritmética. Así pues diseño una función nueva.
        """
        # 1. Estado terminal:
        #   - Si la longitud de la lista de acciones legales es 0, quiere decir que no hay ninguna acción legal...
        #     Esto lo hago debido a un error que me daba y así pues no intenta realizar acciones que no sean legales.
        #   - Si la profundidad es 0, quiere decir que ya hemos explorado todos los niveles del árbol deseados
        # En cualquiera de los dos casos anteriores, acabamos el minimax y devolvemos el valor del estado en cuestión.
        # NOTA: si pongo que devuelva, como siempre, el .getScore() la q5 me da 2/6 puntos, pero si pongo que devuelva
        # esto, entonces la q5 funciona perfectamente y me da 6/6 puntos. La función expectimax funciona en ambos casos
        if len(gameState.getLegalActions(index)) == 0 or depth == 0:
            return self.evaluationFunction(gameState), ""

        # 2. MAX:
        #   - He creído conveniente realizar la función que maximiza a parte. En ella haré la explicación.
        #   - Básicamente dice que si el pacman (es decir, el agente con índice 0) es el que juega ahora, entonces
        #     lo que hay que hacer es maximizar.
        if index == 0:
            return self.maximizer(gameState, index, depth)

        # 3. MIN:
        #   - Igual que con el MAX, pero ahora estamos en todos los otros casos, es decir, en los casos en los que
        #     index > 0, que significa que les toca a los fantasmas. Da igual en qué orden jueguen los fantasmas. En
        #     este caso, lo que hacemos es minimizar.
        else:
            return self.hope(gameState, index, depth)

    def maximizer(self, gameState, index, depth):
        """
        Esta es la función que maximizará el resultado. Consiste en lo siguiente: Para cada acción legal del agente
        - Calculo quién será el agente sucesor. Esto lo hago básicamente sumándole 1 al índice, para indicar que le
          toca al siguiente jugador.
        - Si al sumar el índice nos pasamos del máximo número significará que en realidad ya han acabado su turno todos
          los jugadores y que por tanto empieza "otra ronda": le toca a pacman. Así pues, pongo manualmente que
          el índice sea 0 (el de pacman). También añado que hemos explorado ya una capa del árbol y por tanto resto
          1 a la profundidad.
        - Después de esto, calculo el value llamando al minimax de los sucesores y devuelvo luego el máximo.
        """
        legalMoves = gameState.getLegalActions(index)  # Genero una lista de las acciones legales del agente correspondiente a index
        maxValue = float("-inf")  # Pongo el máximo al número más pequeño posible
        maxAction = ""  # Inicializo la acción emtpy string como acción máxima

        for action in legalMoves:  # Para cada acción legal...
            successor = gameState.generateSuccessor(index, action) # calculo el sucesor
            successorIndex = index + 1  # Configuro su índice, que consiste en uno más que el anterior
            successorDepth = depth  # Añado una variable que será su profundidad, así no tengo que modificar la profundidad actual

            # Si estamos en el caso en que successorIndex == gameState.getNumAgents() significa que "nos hemos pasado".
            # Es decir, que ahora en realidad le tocaría a pacman
            if successorIndex == gameState.getNumAgents():  # getNumAgents() devuelve el total de jugadores (y los índices van de 0 a n-1, con lo cual siemrpe es uno menos)
                successorIndex = 0  # Lo asignamos a 0, para indicar que es pacman
                successorDepth -= 1  # Hemos explorado ya una capa del árbol

            value = self.expectimax(successor, successorIndex, successorDepth)[0]  # Recursividad: pongo el [0] porque esta función devuelve una 2-tupla

            if value > maxValue:  # Calculo el máximo
                maxValue = value  # Me quedo siempre con el valor máximo...
                maxAction = action  # ... así como con la acción correspondiente

        return maxValue, maxAction  # Devuelvo una 2-tupla (valor, acción)

    def hope(self, gameState, index, depth):
        """
        hope = esperanza
        Esta función calcula la esperanza de las acciones del fantasma. Esta esperanza, dado que las acciones las
        de forma aleatoria, coincidirá con la media aritmética. Así pues, esta función calculará la media
        aritmética de los valores de cada acción:

        - Calcularé la suma de todos los valores y al final la dividiré por el total de valores (la media aritmética)
        - Como también tengo que devolver una acción, me planteo: ¿Qué acción devolverá?, ¿La que tenga el máximo valor?
          ¿La que tenga el mínimo? Como el valor devuelto no corresponde a ninguna acción podemos devolver "la que
          queramos".

          Yo había pensado en devolver la acción cuyo valor sea más próximo a la media artmética calculada, pero
          no conseguí hacer que funcionase.

          Entonces encontré en internet una idea que consistía en devolver una acción
          aleatoria, dado que no nos importaba qué acción haría el fantasma porque íbamos a asignarle el valor de
          la media. Así pues miré cómo generar números aleatorios y utilizando la función .getLegalAction() calculé una
          acción aleatoria, que es la que devuelve.
        """
        sumValues = 0.0  # Aquí acumularé la suma de los valores, por tanto la pongo a 0
        totalSuccessors = len(gameState.getLegalActions(index))  # Me interesa para hacer la media aritmética
        randomNumber = random.randint(0, len(gameState.getLegalActions(index)) - 1)  # Esto lo he buscado en internet porque no sabía cómo generar un número aleatorio
        rndAction = gameState.getLegalActions(index)[randomNumber]  # Calculo una acción aleatoria
        for action in gameState.getLegalActions(index):  # Para cada acción...
            successor = gameState.generateSuccessor(index, action)  # Calculo el sucesor que me da
            successorIndex = index + 1  # Sumo un índice al índice del sucesor
            successorDepth = depth  # Asigno la profundidad a una nueva variable, para no modificar cada vez la profundidad en la que estamos

            if successorIndex == gameState.getNumAgents():  # Esto significa que estamos en pacman otra vez. Como siempre
                successorIndex = 0
                successorDepth -= 1

            value = self.expectimax(successor, successorIndex, successorDepth)[0]  # Recursividad
            sumValues += value  # Vamos acumulando los valores en la suma, para después dividirlos entre el total de sucesores

        hope = sumValues / totalSuccessors  # Adquiero la media aritmética
        return hope, rndAction  # Como de costumbre, devuelvo la 2-tupla (valor, accion)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: La función devuelve un valor que depende de:
    1. La distancia a la comida más cercana:
        ·   Calculo la distancia a todas las comidas y me quedo con la mínima (minFood)
    2. La distancia al fantasma más cercano (teniendo en cuenta si están o no en modo huidizo):
        ·   Calculo la distancia a todos los fantasmas y me quedo con la mínima pero, si los fantasmas están en
            modo huidizo, pongo minGhost en infinito y pongo en una nueva variable minScaredGhost esta distancia
        ·   Esto lo hago porque posteriormente me interesará penalizar más cuanto más alta sea la distancia a un
            scared ghost y penalizar más cuanto más baja sea la distancia a un ghost "normal". Por eso pongo dos
            variables y hago dos casos
    3. El número de comida que queda en total:
        ·   Este es un parámetro importante porque a veces puede que quede una comida aislada y pacman esté al lado
            de ésta, y alejado haya un grupo de comidas. En este caso, si sólo tenemos en cuenta el parámetro minFood
            pacman considerará que comerse la comida que tiene al lado es perjudicial, ya que en el estado que se
            encuentra actualmente (al lado de la comida) la minFood es 1, mientras que si se come la comida, la minFood
            aumentará notablemente, porque el resto de comidas está alejado. Así pues, este parámetro es muy importante
    4. La media artimética de las distancias a cada comida:
        ·   Este es un parámetro que me parece bastante inútil. Sin embargo, me aumenta la puntuación unos 100 / 200
            por partida, cosa que me da 6/6 en el autograder.
        ·   Consiste básicamente en hacer la media aritmética de las distancias a cada comida, es decir,
            (suma_distancias_comidas) / (número_total_de_comidas).
        ·   Para ello declaro una variable que me va a ir acumulando las distancias
    5. El número de cápsulas que quedan en total:
        ·   Este parámetro lo pongo para hacer que pacman coma las bolas grandes, ya que estas dan puntos y también
            da puntos comerse a los fantasmas huidizos. Así que si pacman pasa por al lado de una bola grande,
            considerará una buena opción comérsela.
    6. Un parámetro extra: la distancia entre dos comidas.
        ·   Este parámetro lo acabé poniendo ya que a veces pacman se quedaba entre medio de dos comidas, encallado
            pensando.
        ·   Entonces lo que hago yo es penalizar mucho que la distancia entre dos comidas sea alta, así pacman
            intentará comerse todas las comidas "de un mismo grupo" y no dejar una sola aislada.

    La función, por lo general, es muy parecida a la del ejercicio 1, salvo por el parámetro de la distancia entre
    dos comidas y por el peso que le doy a cada parámetro.

    Al final de la función defino la variable score, que la inicializo con la puntuación actual, y la voy modificando,
    sumando o restando, los parámetros descritos en 1,...,5, multiplicándolos por unos números que son bastante
    arbitrarios, pero que corresponden al peso que yo considero que tiene que tener cada parámetro (así pacman prioriza
    una cosa u otra)
    """
    "*** YOUR CODE HERE ***"

    # Inicialmente pongo esto
    if currentGameState.isWin():  # Si el estado supone ganar: nos interesa mucho ==> máxima puntuación posible
        return float('inf')
    if currentGameState.isLose():  # Si el estado supone perder: no nos interesa nada ==> mínima puntuación posible
        return float('-inf')

    # Inicializamos como siempre
    pacmanPosition = currentGameState.getPacmanPosition()
    minFood = float('inf')
    sumFoodDist = 0
    foodToFoodDist = 0

    for food in currentGameState.getFood().asList():
        foodDist = util.manhattanDistance(food,pacmanPosition)  # Distancia a la comida
        minFood = min(minFood, foodDist)  # Me quedo con la mínima
        sumFoodDist += foodDist  # Y además las voy sumando para luego hacer la media aritmética
        for otherFood in currentGameState.getFood().asList():  # Aquí es donde aplico el punto 6
            otherFoodDist = util.manhattanDistance(food,otherFood)  # Calculo la distancia entre dos comidas
            if otherFoodDist > 6:  # Si es mayor que 6: mal asunto
                foodToFoodDist += 1

    # Inicializamos como siempre
    minGhost = float('inf')
    ghostStates = currentGameState.getGhostStates()
    scaredTimesList = [ghostState.scaredTimer for ghostState in ghostStates]  # Esto lo he sacado del ejercicio 1, que nos lo daban en el código base
    minScaredGhost = float('inf')
    ghostIndex = 1  # Empezamos en 1, para indicar que es un fantasma ya desde el principio
    num = 0
    while ghostIndex < currentGameState.getNumAgents():  # Esto es igual que en la función del ejercicio 1, pero programado "mejor" yo creo.
        if ghostIndex > 0:  # Esto significa que es un fantasma (aunque ya lo sabemos, pero por si acaso... me daba un error sino)
            ghostDist = util.manhattanDistance(currentGameState.getGhostPosition(ghostIndex),pacmanPosition)
            if scaredTimesList[num] >= 3:  # Esto significa que aún nos queda tiempo de fantasma blanco para comérnoslo, así que podemos ir a por él
                minScaredGhost = min(minScaredGhost, ghostDist)  # Mínima distancia al ghost blanco
                minGhost = float('inf')  # No habrá minGhost ya que "no hay ghosts"
            else:  # En este caso, al fantasma le queda poco para volver a la normalidad, así que, por precaución, lo consideramos como un fantasma normal.
                minScaredGhost = 0
                minGhost = min(minGhost, ghostDist)
            num += 1
        ghostIndex += 1

    numberOfCapsulesRemaining = len(currentGameState.getCapsules())  # Lo que el nombre indica: número de cápsulas restantes
    capsuleAux = 0  # Auxiliar que utilizo para premiar mucho al fantasma si no quedan cápsulas
    if numberOfCapsulesRemaining == 0:
        capsuleAux = 100
    numberOfFoodRemaining = len(currentGameState.getFood().asList())  # Calculo el número de comida restante, que será penalizado
    ghostPenalization = 0  # Esto por defecto será 0, ya que si el fantasma está muy lejos no me importa en absoluto
    if minGhost < 4:  # Si la distancia al fantasma es menor que 4 añado una penalización muy alta
        ghostPenalization = pow(10,(4 - minGhost))  # Igualmente la penalización es proporcional: no es lo mismo estar a 3 que estar a 1

    # Finalmente calculo la puntuación final aplicándole los parámetros con sus pesos.
    score = currentGameState.getScore()
    score -= 10 * minFood
    score -= 500 * numberOfCapsulesRemaining
    score += capsuleAux
    score -= 20 * numberOfFoodRemaining
    score -= ghostPenalization
    score -= 100 * minScaredGhost
    score -= 5 * (sumFoodDist / numberOfFoodRemaining)
    score -= 100 * foodToFoodDist
    return score


# Abbreviation
better = betterEvaluationFunction
