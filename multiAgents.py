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
        La funci??n que he configurado consiste en calcular la m??nima distancia hasta la comida y penalizar
        cuanto m??s grande sea esta, as?? como calcular la m??nima distancia al fantasma m??s cercano y 
        penalizar si ??sta es peque??a.
        Entonces creo un nuevo estado (que esto ven??a dado ya del c??digo base) que ser?? el estado del sucesor
        En este calculo las distancias que he dicho
        Finalmente creo la variable "score" que es la que va a ir acumulando los puntos para devolverlos al final.
        En esta variable sumo (o resto) la penalizaci??n o la bonificaci??n correspondiente.
        He pensado que ser??a ??til utilizar la funci??n 1/x, ya que si x crece, esta funci??n devuelve valores m??s
        peque??os, mientras que si x decrece, devuelve valores mayores. 
        Ahora bien, si x = 0, entonces no se podr??a calcular porque no est?? definida, as?? pues para arreglarlo
        he pensado en utilizar la funci??n 1/(x+1), que si x es cero devolver?? el valor 1.
        '''
        score = successorGameState.getScore() # Variable en la que acumular?? los puntos
        foodList = newFood.asList()  # Convierte la "matriz" de true/false en una lista con las posiciones en las que hay un true
        closestFoodDist = 9999999999999 # Asigno un valor muy alto, pero que no sea infinito porque me daba problemas
        for food in foodList: # Para cada comida...
            foodDist = util.manhattanDistance(food, newPos) # Calculo su distancia Manhattan desde Pacman
            if foodDist < closestFoodDist: # Hago el m??nimo de estas distancias
                closestFoodDist = foodDist
        closestGhostDist = 9999999999999 # Asigno un valor muy alto, igual que con la comida
        for ghost in newGhostStates: # Para cada fantasma...
            ghostPos = ghost.getPosition() # Adquiero su posici??n
            ghostDist = util.manhattanDistance(ghostPos, newPos) # Calculo la distancia desde Pacman
            if ghostDist < closestGhostDist: # Me quedo con la m??nima
                closestGhostDist = ghostDist
        # A continuaci??n, me planteo que si los fantasmas est??n en modo huidizo, entonces no interesa hu??r de ellos, as?? que podemos ignorarlos
        for ghostTime in newScaredTimes: # newScaredTimes contiene una lista de n??meros que represenetan cuantos
                                         # frames le queda a cada fantasma de ser blanco.
            if ghostTime < 3:  # Si le queda poco tiempo lo tratamos como fantasma normal, porque sino hay mucho riesgo
                # En caso de que el fantasma sea "normal" o est?? a punto de serlo, penalizamos por su proximidad
                score = score + (1.0 / (closestFoodDist + 1.0)) - (1.0 / (closestGhostDist + 1.0))
            else: # En caso de que el fantasma est?? en modo huidizo, no hace falta alejarnos de ??l, as?? pues lo ignoramos.
                score = score + (1.0 / (closestFoodDist + 1.0))

        # Finalmente, he incluido una penalizaci??n muy alta si se da el caso que el fantasma est?? muy muy cerca de Pacman
        # De esta manera, si el Pacman ve que un movimiento le har?? estar a una distancia menor que 2 de un fantasma,
        # no tomar?? nunca la decisi??n de realizarla porque sino recibir?? un score super bajo.
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
        Devolver?? un valor en tipo 2-tupla: (valor, acci??n)

        Hay tres casos:
        1. Estado terminal: ya se ha acabado la profundidad, o bien no quedan acciones legales
        2. Agente maximizador: le toca a pacman y por tanto hay que maximizar la puntuaci??n
        3. Agentes minimizadores: le toca a algunfantasma y por tanto hay que minimizar la puntuaci??n
        """
        # 1. Estado terminal:
        #   - Si la longitud de la lista de acciones legales es 0, quiere decir que no hay ninguna acci??n legal...
        #     Esto lo hago debido a un error que me daba y as?? pues no intenta realizar acciones que no sean legales.
        #   - Si la profundidad es 0, quiere decir que ya hemos explorado todos los niveles del ??rbol deseados
        # En cualquiera de los dos casos anteriores, acabamos el minimax y devolvemos el valor del estado en cuesti??n.
        if len(gameState.getLegalActions(index)) == 0 or depth == 0:
            return gameState.getScore(), ""

        # 2. MAX:
        #   - He cre??do conveniente realizar la funci??n que maximiza a parte. En ella har?? la explicaci??n.
        #   - B??sicamente dice que si el pacman (es decir, el agente con ??ndice 0) es el que juega ahora, entonces
        #     lo que hay que hacer es maximizar.
        if index == 0:
            return self.maximizer(gameState, index, depth)

        # 3. MIN:
        #   - Igual que con el MAX, pero ahora estamos en todos los otros casos, es decir, en los casos en los que
        #     index > 0, que significa que les toca a los fantasmas. Da igual en qu?? orden jueguen los fantasmas. En
        #     este caso, lo que hacemos es minimizar.
        else:
            return self.minimizer(gameState, index, depth)

    def maximizer(self, gameState, index, depth):
        """
        Esta es la funci??n que maximizar?? el resultado. Consiste en lo siguiente: Para cada acci??n legal del agente
        - Calculo qui??n ser?? el agente sucesor. Esto lo hago b??sicamente sum??ndole 1 al ??ndice, para indicar que le
          toca al siguiente jugador.
        - Si al sumar el ??ndice nos pasamos del m??ximo n??mero significar?? que en realidad ya han acabado su turno todos
          los jugadores y que por tanto empieza "otra ronda": le toca a pacman. As?? pues, pongo manualmente que
          el ??ndice sea 0 (el de pacman). Tambi??n a??ado que hemos explorado ya una capa del ??rbol y por tanto resto
          1 a la profundidad.
        - Despu??s de esto, calculo el value llamando al minimax de los sucesores y devuelvo luego el m??ximo.
        """
        legalMoves = gameState.getLegalActions(index)  # Genero una lista de las acciones legales del agente correspondiente a index
        maxValue = float("-inf")  # Pongo el m??ximo al n??mero m??s peque??o posible
        maxAction = ""  # Inicializo la acci??n emtpy string como acci??n m??xima

        for action in legalMoves:  # Para cada acci??n legal...
            successor = gameState.generateSuccessor(index, action) # calculo el sucesor
            successorIndex = index + 1  # Configuro su ??ndice, que consiste en uno m??s que el anterior
            successorDepth = depth  # A??ado una variable que ser?? su profundidad, as?? no tengo que modificar la profundidad actual

            # Si estamos en el caso en que successorIndex == gameState.getNumAgents() significa que "nos hemos pasado".
            # Es decir, que ahora en realidad le tocar??a a pacman
            if successorIndex == gameState.getNumAgents():  # getNumAgents() devuelve el total de jugadores (y los ??ndices van de 0 a n-1, con lo cual siemrpe es uno menos)
                successorIndex = 0  # Lo asignamos a 0, para indicar que es pacman
                successorDepth -= 1  # Hemos explorado ya una capa del ??rbol

            value = self.minimax(successor, successorIndex, successorDepth)[0]  # Recursividad: pongo el [0] porque esta funci??n devuelve una 2-tupla

            if value > maxValue:  # Calculo el m??ximo
                maxValue = value  # Me quedo siempre con el valor m??ximo...
                maxAction = action  # ... as?? como con la acci??n correspondiente

        return maxValue, maxAction  # Devuelvo una 2-tupla (valor, acci??n)

    def minimizer(self, gameState, index, depth):
        """
        Esta funci??n minimizar?? el resultado. Es bastante an??loga a la que maximiza, pero cambiando max por min. No vuelvo
        a hacer toda la explicaci??n porque es la misma:
        - En vez de inicializar en -inf, inicializamos en +inf
        - Lo de si successorIndex == gameState.getNumAgents() es igual
        - En vez de buscar el m??ximo, buscamos ahora el m??nimo
        """
        legalMoves = gameState.getLegalActions(index)
        minValue = float("inf")
        minAction = ""

        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action)
            successorIndex = index + 1
            successorDepth = depth

            # Si estamos en el caso en que successorIndex == gameState.getNumAgents() significa que "nos hemos pasado".
            # Es decir, que ahora en realidad le tocar??a a pacman
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
        # Formato del resultado: 2-tupla del tipo (valor, acci??n). Queremos s??lo la acci??n, por tanto...
        result = self.alphaBeta(gameState, 0, self.depth, float('-inf'), float('+inf'))
        # ...devolvemos s??lo la segunda componente
        return result[1]

    def alphaBeta(self, gameState, index, depth, alpha, beta):
        """
        La funci??n minimax con poda alpha-beta no difiere mucho de la funci??n minimax a secas. Es por ello que he
        realizado un "copy-paste" de las funciones del minimax, y he a??adido alguna cosa para realizar la poda:

        Como siempre, tenemos tres casos:
        1. Estado terminal: o bien ya no quedan acciones legales, o bien hemso explorado todas las capas deseadas
        2. Le toca a pacman (index = 0) y por tanto hay que maximizar
        3. Le toca a alg??n fantasma (index > 0) y por tanto hay que minimizar
        """
        # 1. Estado terminal:
        #   - Si la longitud de la lista de acciones legales es 0, quiere decir que no hay ninguna acci??n legal...
        #     Esto lo hago debido a un error que me daba y as?? pues no intenta realizar acciones que no sean legales.
        #   - Si la profundidad es 0, quiere decir que ya hemos explorado todos los niveles del ??rbol deseados
        # En cualquiera de los dos casos anteriores, acabamos el minimax y devolvemos el valor del estado en cuesti??n.
        if len(gameState.getLegalActions(index)) == 0 or depth == 0:
            return gameState.getScore(), ""

        # 2. MAX:
        #   - He cre??do conveniente realizar la funci??n que maximiza a parte. En ella har?? la explicaci??n.
        #   - B??sicamente dice que si el pacman (es decir, el agente con ??ndice 0) es el que juega ahora, entonces
        #     lo que hay que hacer es maximizar.
        if index == 0:
            return self.maximizer(gameState, index, depth, alpha, beta)

        # 3. MIN:
        #   - Igual que con el MAX, pero ahora estamos en todos los otros casos, es decir, en los casos en los que
        #     index > 0, que significa que les toca a los fantasmas. Da igual en qu?? orden jueguen los fantasmas. En
        #     este caso, lo que hacemos es minimizar.
        else:
            return self.minimizer(gameState, index, depth, alpha, beta)

    def maximizer(self, gameState, index, depth, alpha, beta):
        """
        Maximiza, teniendo en cuenta que el m??ximo sea inferior a beta. Si supera a beta, paramos:
        - Calculo qui??n ser?? el agente sucesor. Esto lo hago b??sicamente sum??ndole 1 al ??ndice, para indicar que le
          toca al siguiente jugador.

        - Si al sumar el ??ndice nos pasamos del m??ximo n??mero significar?? que en realidad ya han acabado su turno todos
          los jugadores y que por tanto empieza "otra ronda": le toca a pacman. As?? pues, pongo manualmente que
          el ??ndice sea 0 (el de pacman). Tambi??n a??ado que hemos explorado ya una capa del ??rbol y por tanto resto
          1 a la profundidad.

        - Despu??s de esto, calculo el value llamando al minimax de los sucesores y devuelvo luego el m??ximo. Al realizar
          este m??ximo, calculo alpha como el m??ximo entre alpha y el m??ximo calculado. As?? guardo en alpha el valor
          m??ximo.

        - Antes de esto, de hecho, se realiza la poda (si se ha de realizar). Esta poda se dar?? si el m??ximo que estoy
          calculando ya no supera el m??nimo (establecido por beta) y devolveremos directamente el resultado, saliendo
          del bucle "for" ya que no valdr?? la pena seguir: si un valor ya supera a beta, como nos quedaremos con el m??ximo
          seguro que superar?? a beta y esto no nos interesa: no hace falta seguir.
        """
        legalMoves = gameState.getLegalActions(index)  # Genero una lista de las acciones legales del agente correspondiente a index
        maxValue = float("-inf")  # Pongo el m??ximo al n??mero m??s peque??o posible
        maxAction = ""  # Inicializo la acci??n emtpy string como acci??n m??xima

        for action in legalMoves:  # Para cada acci??n legal...
            successor = gameState.generateSuccessor(index, action) # calculo el sucesor
            successorIndex = index + 1  # Configuro su ??ndice, que consiste en uno m??s que el anterior
            successorDepth = depth  # A??ado una variable que ser?? su profundidad, as?? no tengo que modificar la profundidad actual

            # Si estamos en el caso en que successorIndex == gameState.getNumAgents() significa que "nos hemos pasado".
            # Es decir, que ahora en realidad le tocar??a a pacman
            if successorIndex == gameState.getNumAgents():  # getNumAgents() devuelve el total de jugadores (y los ??ndices van de 0 a n-1, con lo cual siemrpe es uno menos)
                successorIndex = 0  # Lo asignamos a 0, para indicar que es pacman
                successorDepth -= 1  # Hemos explorado ya una capa del ??rbol

            value = self.alphaBeta(successor, successorIndex, successorDepth, alpha, beta)[0]  # Recursividad: pongo el [0] porque esta funci??n devuelve una 2-tupla

            if value > maxValue:  # Calculo el m??ximo
                maxValue = value  # Me quedo siempre con el valor m??ximo...
                maxAction = action  # ... as?? como con la acci??n correspondiente
            if maxValue > beta: # PODAMOS
                return maxValue, maxAction
            alpha = max(alpha,maxValue) # Vamos calculando, al mismo tiempo, el m??ximo alpha, que nos servir?? para podar en el m??nimo.
        return maxValue, maxAction  # Devuelvo una 2-tupla (valor, acci??n)

    def minimizer(self, gameState, index, depth, alpha, beta):
        """
        Esta funci??n minimizar?? el resultado. Es bastante an??loga a la que maximiza, pero cambiando max por min. No vuelvo
        a hacer toda la explicaci??n porque es la misma:
        - En vez de inicializar en -inf, inicializamos en +inf
        - Lo de si successorIndex == gameState.getNumAgents() es igual
        - En vez de buscar el m??ximo, buscamos ahora el m??nimo.
        - Si el valor se pasa (inferiormente) de alpha, significa que no vamos a obtener un m??nimo por encima de alpha,
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
            # Es decir, que ahora en realidad le tocar??a a pacman
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
        # Como siempre, el resultado es en forma de 2-tupla (valor, acci??n) y nosotros queremos s??lo la acci??n.
        return self.expectimax(gameState, 0, self.depth)[1]

    def expectimax(self, gameState, index, depth):
        """
        Es bastante parecido al minimax pero con una diferencia: el minimax asume que el contrincante
        jugar?? de la mejor manera posible. En el caso de los fantasmas esto no es cierto del to do, pues los fantasmas
        actuar??n de foma aleatoria (seg??n tengo entendido) y as?? pues, no siempre escoger??n la mejor opci??n

        Lo que hace el expectimax es, pues, no intentar minimizar la acci??n del contrincante, sino intentar tomar
        la probabilidad de que el fantasma haga tal movimiento. Es decir, la esperanza. Como el fantasma toma
        las acciones de manera aleatoria, la esperanza es igual a la media aritm??tica de todos los valores posibles

        Como de costumbre, tenemos tres casos posibles:
        1. Estado terminal
            Este caso se da si ya hemos explorado todas las capas deseadas o bien si no hay acciones legales que se
            puedan realizar. En este caso devolvemos la puntuaci??n del estado.
        2. Le toca a pacman:
            Este caso se da si le toca a pacman, es decir, si el ??ndice es 0. En este caso pacman seguir??, como siempre
            intentando maximizar su jugada. Por tanto, ejecuta la funci??n maxValue que es una copia de la del minimax
        3. Le toca a un fantasma:
            Este tercer caso se dar?? si el ??nidce es mayor que cero: esto indica que le toca a un fantasma.
            Aqu?? es lo que cambia respecto a la funci??n minimax: en vez de minimizar, calcula la esperanza de
            escoger tal valor, que coincide con la media aritm??tica. As?? pues dise??o una funci??n nueva.
        """
        # 1. Estado terminal:
        #   - Si la longitud de la lista de acciones legales es 0, quiere decir que no hay ninguna acci??n legal...
        #     Esto lo hago debido a un error que me daba y as?? pues no intenta realizar acciones que no sean legales.
        #   - Si la profundidad es 0, quiere decir que ya hemos explorado todos los niveles del ??rbol deseados
        # En cualquiera de los dos casos anteriores, acabamos el minimax y devolvemos el valor del estado en cuesti??n.
        # NOTA: si pongo que devuelva, como siempre, el .getScore() la q5 me da 2/6 puntos, pero si pongo que devuelva
        # esto, entonces la q5 funciona perfectamente y me da 6/6 puntos. La funci??n expectimax funciona en ambos casos
        if len(gameState.getLegalActions(index)) == 0 or depth == 0:
            return self.evaluationFunction(gameState), ""

        # 2. MAX:
        #   - He cre??do conveniente realizar la funci??n que maximiza a parte. En ella har?? la explicaci??n.
        #   - B??sicamente dice que si el pacman (es decir, el agente con ??ndice 0) es el que juega ahora, entonces
        #     lo que hay que hacer es maximizar.
        if index == 0:
            return self.maximizer(gameState, index, depth)

        # 3. MIN:
        #   - Igual que con el MAX, pero ahora estamos en todos los otros casos, es decir, en los casos en los que
        #     index > 0, que significa que les toca a los fantasmas. Da igual en qu?? orden jueguen los fantasmas. En
        #     este caso, lo que hacemos es minimizar.
        else:
            return self.hope(gameState, index, depth)

    def maximizer(self, gameState, index, depth):
        """
        Esta es la funci??n que maximizar?? el resultado. Consiste en lo siguiente: Para cada acci??n legal del agente
        - Calculo qui??n ser?? el agente sucesor. Esto lo hago b??sicamente sum??ndole 1 al ??ndice, para indicar que le
          toca al siguiente jugador.
        - Si al sumar el ??ndice nos pasamos del m??ximo n??mero significar?? que en realidad ya han acabado su turno todos
          los jugadores y que por tanto empieza "otra ronda": le toca a pacman. As?? pues, pongo manualmente que
          el ??ndice sea 0 (el de pacman). Tambi??n a??ado que hemos explorado ya una capa del ??rbol y por tanto resto
          1 a la profundidad.
        - Despu??s de esto, calculo el value llamando al minimax de los sucesores y devuelvo luego el m??ximo.
        """
        legalMoves = gameState.getLegalActions(index)  # Genero una lista de las acciones legales del agente correspondiente a index
        maxValue = float("-inf")  # Pongo el m??ximo al n??mero m??s peque??o posible
        maxAction = ""  # Inicializo la acci??n emtpy string como acci??n m??xima

        for action in legalMoves:  # Para cada acci??n legal...
            successor = gameState.generateSuccessor(index, action) # calculo el sucesor
            successorIndex = index + 1  # Configuro su ??ndice, que consiste en uno m??s que el anterior
            successorDepth = depth  # A??ado una variable que ser?? su profundidad, as?? no tengo que modificar la profundidad actual

            # Si estamos en el caso en que successorIndex == gameState.getNumAgents() significa que "nos hemos pasado".
            # Es decir, que ahora en realidad le tocar??a a pacman
            if successorIndex == gameState.getNumAgents():  # getNumAgents() devuelve el total de jugadores (y los ??ndices van de 0 a n-1, con lo cual siemrpe es uno menos)
                successorIndex = 0  # Lo asignamos a 0, para indicar que es pacman
                successorDepth -= 1  # Hemos explorado ya una capa del ??rbol

            value = self.expectimax(successor, successorIndex, successorDepth)[0]  # Recursividad: pongo el [0] porque esta funci??n devuelve una 2-tupla

            if value > maxValue:  # Calculo el m??ximo
                maxValue = value  # Me quedo siempre con el valor m??ximo...
                maxAction = action  # ... as?? como con la acci??n correspondiente

        return maxValue, maxAction  # Devuelvo una 2-tupla (valor, acci??n)

    def hope(self, gameState, index, depth):
        """
        hope = esperanza
        Esta funci??n calcula la esperanza de las acciones del fantasma. Esta esperanza, dado que las acciones las
        de forma aleatoria, coincidir?? con la media aritm??tica. As?? pues, esta funci??n calcular?? la media
        aritm??tica de los valores de cada acci??n:

        - Calcular?? la suma de todos los valores y al final la dividir?? por el total de valores (la media aritm??tica)
        - Como tambi??n tengo que devolver una acci??n, me planteo: ??Qu?? acci??n devolver???, ??La que tenga el m??ximo valor?
          ??La que tenga el m??nimo? Como el valor devuelto no corresponde a ninguna acci??n podemos devolver "la que
          queramos".

          Yo hab??a pensado en devolver la acci??n cuyo valor sea m??s pr??ximo a la media artm??tica calculada, pero
          no consegu?? hacer que funcionase.

          Entonces encontr?? en internet una idea que consist??a en devolver una acci??n
          aleatoria, dado que no nos importaba qu?? acci??n har??a el fantasma porque ??bamos a asignarle el valor de
          la media. As?? pues mir?? c??mo generar n??meros aleatorios y utilizando la funci??n .getLegalAction() calcul?? una
          acci??n aleatoria, que es la que devuelve.
        """
        sumValues = 0.0  # Aqu?? acumular?? la suma de los valores, por tanto la pongo a 0
        totalSuccessors = len(gameState.getLegalActions(index))  # Me interesa para hacer la media aritm??tica
        randomNumber = random.randint(0, len(gameState.getLegalActions(index)) - 1)  # Esto lo he buscado en internet porque no sab??a c??mo generar un n??mero aleatorio
        rndAction = gameState.getLegalActions(index)[randomNumber]  # Calculo una acci??n aleatoria
        for action in gameState.getLegalActions(index):  # Para cada acci??n...
            successor = gameState.generateSuccessor(index, action)  # Calculo el sucesor que me da
            successorIndex = index + 1  # Sumo un ??ndice al ??ndice del sucesor
            successorDepth = depth  # Asigno la profundidad a una nueva variable, para no modificar cada vez la profundidad en la que estamos

            if successorIndex == gameState.getNumAgents():  # Esto significa que estamos en pacman otra vez. Como siempre
                successorIndex = 0
                successorDepth -= 1

            value = self.expectimax(successor, successorIndex, successorDepth)[0]  # Recursividad
            sumValues += value  # Vamos acumulando los valores en la suma, para despu??s dividirlos entre el total de sucesores

        hope = sumValues / totalSuccessors  # Adquiero la media aritm??tica
        return hope, rndAction  # Como de costumbre, devuelvo la 2-tupla (valor, accion)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: La funci??n devuelve un valor que depende de:
    1. La distancia a la comida m??s cercana:
        ??   Calculo la distancia a todas las comidas y me quedo con la m??nima (minFood)
    2. La distancia al fantasma m??s cercano (teniendo en cuenta si est??n o no en modo huidizo):
        ??   Calculo la distancia a todos los fantasmas y me quedo con la m??nima pero, si los fantasmas est??n en
            modo huidizo, pongo minGhost en infinito y pongo en una nueva variable minScaredGhost esta distancia
        ??   Esto lo hago porque posteriormente me interesar?? penalizar m??s cuanto m??s alta sea la distancia a un
            scared ghost y penalizar m??s cuanto m??s baja sea la distancia a un ghost "normal". Por eso pongo dos
            variables y hago dos casos
    3. El n??mero de comida que queda en total:
        ??   Este es un par??metro importante porque a veces puede que quede una comida aislada y pacman est?? al lado
            de ??sta, y alejado haya un grupo de comidas. En este caso, si s??lo tenemos en cuenta el par??metro minFood
            pacman considerar?? que comerse la comida que tiene al lado es perjudicial, ya que en el estado que se
            encuentra actualmente (al lado de la comida) la minFood es 1, mientras que si se come la comida, la minFood
            aumentar?? notablemente, porque el resto de comidas est?? alejado. As?? pues, este par??metro es muy importante
    4. La media artim??tica de las distancias a cada comida:
        ??   Este es un par??metro que me parece bastante in??til. Sin embargo, me aumenta la puntuaci??n unos 100 / 200
            por partida, cosa que me da 6/6 en el autograder.
        ??   Consiste b??sicamente en hacer la media aritm??tica de las distancias a cada comida, es decir,
            (suma_distancias_comidas) / (n??mero_total_de_comidas).
        ??   Para ello declaro una variable que me va a ir acumulando las distancias
    5. El n??mero de c??psulas que quedan en total:
        ??   Este par??metro lo pongo para hacer que pacman coma las bolas grandes, ya que estas dan puntos y tambi??n
            da puntos comerse a los fantasmas huidizos. As?? que si pacman pasa por al lado de una bola grande,
            considerar?? una buena opci??n com??rsela.
    6. Un par??metro extra: la distancia entre dos comidas.
        ??   Este par??metro lo acab?? poniendo ya que a veces pacman se quedaba entre medio de dos comidas, encallado
            pensando.
        ??   Entonces lo que hago yo es penalizar mucho que la distancia entre dos comidas sea alta, as?? pacman
            intentar?? comerse todas las comidas "de un mismo grupo" y no dejar una sola aislada.

    La funci??n, por lo general, es muy parecida a la del ejercicio 1, salvo por el par??metro de la distancia entre
    dos comidas y por el peso que le doy a cada par??metro.

    Al final de la funci??n defino la variable score, que la inicializo con la puntuaci??n actual, y la voy modificando,
    sumando o restando, los par??metros descritos en 1,...,5, multiplic??ndolos por unos n??meros que son bastante
    arbitrarios, pero que corresponden al peso que yo considero que tiene que tener cada par??metro (as?? pacman prioriza
    una cosa u otra)
    """
    "*** YOUR CODE HERE ***"

    # Inicialmente pongo esto
    if currentGameState.isWin():  # Si el estado supone ganar: nos interesa mucho ==> m??xima puntuaci??n posible
        return float('inf')
    if currentGameState.isLose():  # Si el estado supone perder: no nos interesa nada ==> m??nima puntuaci??n posible
        return float('-inf')

    # Inicializamos como siempre
    pacmanPosition = currentGameState.getPacmanPosition()
    minFood = float('inf')
    sumFoodDist = 0
    foodToFoodDist = 0

    for food in currentGameState.getFood().asList():
        foodDist = util.manhattanDistance(food,pacmanPosition)  # Distancia a la comida
        minFood = min(minFood, foodDist)  # Me quedo con la m??nima
        sumFoodDist += foodDist  # Y adem??s las voy sumando para luego hacer la media aritm??tica
        for otherFood in currentGameState.getFood().asList():  # Aqu?? es donde aplico el punto 6
            otherFoodDist = util.manhattanDistance(food,otherFood)  # Calculo la distancia entre dos comidas
            if otherFoodDist > 6:  # Si es mayor que 6: mal asunto
                foodToFoodDist += 1

    # Inicializamos como siempre
    minGhost = float('inf')
    ghostStates = currentGameState.getGhostStates()
    scaredTimesList = [ghostState.scaredTimer for ghostState in ghostStates]  # Esto lo he sacado del ejercicio 1, que nos lo daban en el c??digo base
    minScaredGhost = float('inf')
    ghostIndex = 1  # Empezamos en 1, para indicar que es un fantasma ya desde el principio
    num = 0
    while ghostIndex < currentGameState.getNumAgents():  # Esto es igual que en la funci??n del ejercicio 1, pero programado "mejor" yo creo.
        if ghostIndex > 0:  # Esto significa que es un fantasma (aunque ya lo sabemos, pero por si acaso... me daba un error sino)
            ghostDist = util.manhattanDistance(currentGameState.getGhostPosition(ghostIndex),pacmanPosition)
            if scaredTimesList[num] >= 3:  # Esto significa que a??n nos queda tiempo de fantasma blanco para com??rnoslo, as?? que podemos ir a por ??l
                minScaredGhost = min(minScaredGhost, ghostDist)  # M??nima distancia al ghost blanco
                minGhost = float('inf')  # No habr?? minGhost ya que "no hay ghosts"
            else:  # En este caso, al fantasma le queda poco para volver a la normalidad, as?? que, por precauci??n, lo consideramos como un fantasma normal.
                minScaredGhost = 0
                minGhost = min(minGhost, ghostDist)
            num += 1
        ghostIndex += 1

    numberOfCapsulesRemaining = len(currentGameState.getCapsules())  # Lo que el nombre indica: n??mero de c??psulas restantes
    capsuleAux = 0  # Auxiliar que utilizo para premiar mucho al fantasma si no quedan c??psulas
    if numberOfCapsulesRemaining == 0:
        capsuleAux = 100
    numberOfFoodRemaining = len(currentGameState.getFood().asList())  # Calculo el n??mero de comida restante, que ser?? penalizado
    ghostPenalization = 0  # Esto por defecto ser?? 0, ya que si el fantasma est?? muy lejos no me importa en absoluto
    if minGhost < 4:  # Si la distancia al fantasma es menor que 4 a??ado una penalizaci??n muy alta
        ghostPenalization = pow(10,(4 - minGhost))  # Igualmente la penalizaci??n es proporcional: no es lo mismo estar a 3 que estar a 1

    # Finalmente calculo la puntuaci??n final aplic??ndole los par??metros con sus pesos.
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
