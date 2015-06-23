/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package teitp2ia;

import java.io.FileReader;
import java.util.Random;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author gabriel
 */
public class guarda {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        		// Lendo os exemplos a partir do arquivo guarda.arff
		FileReader leitor = new FileReader("guarda.arff");
		Instances guarda = new Instances(leitor);
		
		// Definindo o índice do atributo classe (último atributo do conjunto)
		guarda.setClassIndex(guarda.numAttributes() - 1);
		
		// Criando uma nova base com os exemplos embaralhados 
		guarda = guarda.resample(new Random());			
		
		// Abordagem Hold out de validação cruzada 
		Instances baseTeste = guarda.testCV(3, 0); // Obtendo subconjunto para testes
		Instances baseTreino = guarda.trainCV(3, 0); // Obtendo subconjunto para treinamento
		
		// Criando os classificadores que serão avaliados
		IBk knn = new IBk(3); // knn com 3 vizinhos
		IB1 vizinho = new IB1(); // vizinho mais próximo
		
		// Treinando os classificadores instanciados
		knn.buildClassifier(baseTreino);
		vizinho.buildClassifier(baseTreino);
		
		System.out.println("real\tknn\tvizinho"); // imprimindo rótulos para as colunas
		
		for (int e = 0; e < baseTeste.numInstances(); e++) {
			Instance exemplo = baseTeste.instance(e);
			System.out.print(exemplo.classValue()); // imprimindo o valor da classe real do exemplo
			exemplo.setClassMissing(); // removendo informação da classe
			double classe = knn.classifyInstance(exemplo); // resposta do knn
			System.out.print("\t" + classe); // imprimindo resposta do knn
			classe = vizinho.classifyInstance(exemplo); // resposta do vizinho mais próximo
			System.out.println("\t" + classe); // imprimindo resposta do vizinho mais próximo
		}
    }
    
}
