    def prueba(x,theta)
      H = sigmoide(np.dot(X, Theta))
      desa = (H < 0.5)
      desaprobados = len(H(desa))
      aprobados = len(H) - desaprobados
      return aprobados, desaprobados
